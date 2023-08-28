import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

class HalfBandFIR:
    def __init__(self, center_tap, nonzero_first_half):
        N = len(nonzero_first_half)
        self.N_TAPS = 4*N-1
        self.TAPS = np.zeros(self.N_TAPS, dtype=np.int64)
        self.TAPS[:2*N-1:2] = nonzero_first_half
        self.TAPS[2*N-1] = center_tap
        self.TAPS[2*N::2] = nonzero_first_half[::-1]


class FIR0a(HalfBandFIR):
    def __init__(self):
        super().__init__(1024, [-3, 477])
class FIR0(HalfBandFIR):
    def __init__(self):
        super().__init__(2048, [-6, 54, -254, 1230])
class FIR1(HalfBandFIR):
    def __init__(self):
        super().__init__(16384, [-12, 84, -337, 1008, -2693, 10142])
class FIR2(HalfBandFIR):
    def __init__(self):
        super().__init__(65536, [5, -17, 44, -96, 187, -335, 565, -906, 1401, -2112, 3145, -4723, 7415, -13331, 41526])

class DSP48E2_MAC:
    # configured with all registers, WE_B disabled (fixed filter weight)
    # A[0] > A[-1] > A[-2] + > (A[-3] + D[-2]) * > B*(A[-4] + D[-3]) + > C[-2] + B*(A[-5] + D[-4])
    #      |  D[0] > D[-1] / |               B / |      C[0] - C[-1] / |
    #      |       |         |                   |                     |
    #     A1     A2/D       AD                   M                     P
    #
    # total latency from A input to P output: 5 cycles
    # A has 3 more regs than C and 1 more reg than D in its path
    # this can be advantageous for implementing the required delay for the filter when
    #  N_TAPS > N_SAMP_IN
    def __init__(self, weight):
        # initialize regs
        self.B = weight
        self.A1 = 0
        self.A2 = 0
        self.D = 0
        self.AD = 0
        self.C = 0
        self.M = 0
        self.P = 0

    def step(self, A, D, C):
        self.P = self.M + self.C
        self.C = C
        self.M = self.AD * self.B
        self.AD = self.D + self.A2
        self.A2 = self.A1
        self.A1 = A
        self.D = D

class Shiftreg:
    def __init__(self, N_stages):
        self.BYPASS = N_stages < 1
        self.state = [0]*N_stages
        self.Q = 0

    def step(self, D):
        self.state = [D] + self.state[:-1]
        if not(self.BYPASS):
            self.Q = self.state[-1]

class HW_HalfBandFIR:
    def __init__(self, N_samp_in, FIR, debug=False):
        self.DEBUG = debug
        self.N_SAMP_IN = N_samp_in
        self.N_SAMP_OUT = N_samp_in // 2
        self.N_TAPS = FIR.N_TAPS
        self.FILTER_DEPTH = (self.N_TAPS+1)//4 # works for halfband FIR filters due to coefficient symmetry
        self.DSP48s = []
        self.Y = np.zeros(self.N_SAMP_OUT)
        self.CENTER_TAP = FIR.TAPS[(self.N_TAPS+1)//2-1]
        for i in range(self.N_SAMP_OUT):
            self.DSP48s.append([])
            for j in range(self.FILTER_DEPTH):
                self.DSP48s[i].append(DSP48E2_MAC(FIR.TAPS[2*j]))
        # make appropriate shift registers and calculate needed delay per input
        self.SHIFTREGS_ODD = [None for i in range(self.N_SAMP_OUT)]
        self.SHIFTREGS_EVEN = [[] for i in range(self.N_SAMP_OUT)]
        even_input_delays = np.zeros((self.N_SAMP_OUT, self.FILTER_DEPTH, 2), dtype=int)
        odd_input_delays = np.zeros(self.N_SAMP_OUT, dtype=int)
        # just used to assist with simulation/verification
        self.TOTAL_LATENCY = 0
        for i in reversed(range(self.N_SAMP_OUT)):
            # C input delay for middle weight
            odd_input_delays[i] =  -(((-self.N_TAPS)//2 + 1 + 2 * i) // self.N_SAMP_IN) + 3 # extra three stages to match C input with A input of DSP48
            # additional delay due to cascading DSP48s
            # note that this can be reduced by using a tree structure (which trades off using more DSP48s for fewer shift register stages)
            stage_delay = 0
            for j in reversed(range(self.FILTER_DEPTH)):
                # D input
                even_input_delays[i,j,0] = -((2 * (i - j)) // self.N_SAMP_IN) + stage_delay + 1 # need one extra delay stage to match D input with A input of DSP48
                # A input
                even_input_delays[i,j,1] = -((2 * (i + j + 1) - self.N_TAPS - 1) // self.N_SAMP_IN) + stage_delay 
                stage_delay += 2
        m = np.min(even_input_delays)
        even_input_delays -= m
        odd_input_delays -= m
        self.TOTAL_LATENCY = even_input_delays[0,0,0] + 4 + 1
        for i in range(self.N_SAMP_OUT):
            self.SHIFTREGS_ODD[i] = Shiftreg(odd_input_delays[i])
            self.SHIFTREGS_EVEN[i] = [None for j in range(2*self.FILTER_DEPTH)]
            for j in range(self.FILTER_DEPTH):
                #d_max = even_input_delays[i,j,0] + 5 + 2*j - 1
                #if d_max > self.TOTAL_LATENCY:
                #    self.TOTAL_LATENCY = d_max
                self.SHIFTREGS_EVEN[i][2*j] = Shiftreg(even_input_delays[i,j,0]) # D input
                self.SHIFTREGS_EVEN[i][2*j+1] = Shiftreg(even_input_delays[i,j,1]) # A input
        if self.DEBUG:
            print(f'N_TAPS = {self.N_TAPS}, FILTER_DEPTH = {self.FILTER_DEPTH}, TOTAL_LATENCY = {self.TOTAL_LATENCY}')
            print(f'even_input_delays = {even_input_delays}')
            print(f'odd_input_delays = {odd_input_delays}')
            for i in range(self.N_SAMP_OUT):
                for j in range(self.FILTER_DEPTH):
                    print(f'(i,j) = {i,j}, (xD,xA,xC) = {(2 * (i - j)) % self.N_SAMP_IN, (2 * (i + j + 1) - self.N_TAPS - 1) % self.N_SAMP_IN, ((-self.N_TAPS)//2 + 1 + 2 * i) % self.N_SAMP_IN}')

    def step(self, x_in):
        for i in range(self.N_SAMP_OUT):
            self.Y[i] = self.DSP48s[i][0].P
            for j in range(self.FILTER_DEPTH):
                # DSP48.step(A, D, C)
                x1 = x_in[(2 * (i - j)) % self.N_SAMP_IN] # D input
                x2 = x_in[(2 * (i + j + 1) - self.N_TAPS - 1) % self.N_SAMP_IN] # A input
                D = x1 if self.SHIFTREGS_EVEN[i][2*j].BYPASS else self.SHIFTREGS_EVEN[i][2*j].Q
                A = x2 if self.SHIFTREGS_EVEN[i][2*j+1].BYPASS else self.SHIFTREGS_EVEN[i][2*j+1].Q
                C = self.DSP48s[i][j+1].P if j < self.FILTER_DEPTH - 1 else self.SHIFTREGS_ODD[i].Q
                self.DSP48s[i][j].step(A, D, C)
                self.SHIFTREGS_EVEN[i][2*j].step(x1)
                self.SHIFTREGS_EVEN[i][2*j+1].step(x2)
            self.SHIFTREGS_ODD[i].step(self.CENTER_TAP*x_in[((-self.N_TAPS)//2 + 1 + 2 * i) % self.N_SAMP_IN])

if __name__ == '__main__':
    # generate signal at 4 GS/s
    fs_in = 4.096e9
    T_in = 1e-6
    N_in = int(fs_in*T_in)
    t_in = np.linspace(0,T_in,N_in) # s
    #x = np.sin(2*np.pi*5e6*t_in) #+ 0.2*np.random.randn(N_in) + np.sin(2*np.pi*fs_in/3*t_in)
    #x = np.sin(2*np.pi*fs_in/3*t_in)
    #x = 0.2*np.random.randn(N_in)
    x = np.sin(2*np.pi*5e6*t_in) + np.sin(2*np.pi*fs_in/3*t_in)

    for k in range(3):
        if k == 0:
            N_SAMP_ADC = 4 # samples per cycle from ADC
        elif k == 1:
            N_SAMP_ADC = 8 # samples per cycle from ADC
        else:
            N_SAMP_ADC = 16 # samples per cycle from ADC

        # initialize FIR taps
        firs = [FIR0a(), FIR0(), FIR1(), FIR2()]
        debug = [False, False, False, False]
        plot = [True, True, True, True]

        # apply filter in software to signal
        sw_sim_result = [lfilter(fir.TAPS, 1.0, x)[::2] for fir in firs]
        sw_poly_sim_result = [lfilter(fir.TAPS[0::2], 1.0, x[0::2]) + lfilter(fir.TAPS[1::2] + [0], 1.0, np.concatenate(([0], x[1:-2:2]))) for fir in firs]

        # simulate hardware for filter
        hw_sim_firs = [HW_HalfBandFIR(N_SAMP_ADC, fir, dbg) for fir, dbg in zip(firs,debug)]
        hw_sim_result = np.zeros((len(firs),N_in//2))
        for cycle in range(N_in//N_SAMP_ADC):
            for i in range(len(firs)):
                if cycle >= hw_sim_firs[i].TOTAL_LATENCY:
                    cycle0 = cycle - hw_sim_firs[i].TOTAL_LATENCY
                    hw_sim_result[i,(N_SAMP_ADC//2)*cycle0:(N_SAMP_ADC//2)*(cycle0+1)] = hw_sim_firs[i].Y
                hw_sim_firs[i].step(x[N_SAMP_ADC*cycle:N_SAMP_ADC*(cycle+1)])

        # plot results
        for i in range(len(firs)):
            if not plot[i]:
                continue
            fig, ax = plt.subplots(3,1,sharex=True)
            ax[0].plot(t_in, x, label='input')
            ax[1].plot(t_in[::2], sw_sim_result[i], label=f'fir{i} software impl')
            ax[1].plot(t_in[::2], sw_poly_sim_result[i], label=f'fir{i} software (polyphase) impl')
            ax[1].plot(t_in[::2], hw_sim_result[i], label=f'fir{i} hardware impl')
            t_in_trim = t_in[:-hw_sim_firs[i].TOTAL_LATENCY*N_SAMP_ADC:2]
            fir_sw_poly_trim = sw_poly_sim_result[i][:-hw_sim_firs[i].TOTAL_LATENCY*(N_SAMP_ADC//2)]
            fir_hw_trim = hw_sim_result[i,:-hw_sim_firs[i].TOTAL_LATENCY*(N_SAMP_ADC//2)]
            ax[2].plot(t_in_trim, fir_sw_poly_trim - fir_hw_trim, label='error')
            for j in range(len(ax)):
                ax[j].legend()
            fig.suptitle(f'firidx = {i}, N_TAPS = {firs[i].N_TAPS}, N_SAMP_ADC = {N_SAMP_ADC}')

    plt.show()
