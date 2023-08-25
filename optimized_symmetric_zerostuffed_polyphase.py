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
    def __init__(self, N_samp_in, FIR):
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
        max_delay = (self.N_TAPS - 2) // self.N_SAMP_IN
        for i in reversed(range(self.N_SAMP_OUT)):
            # process each output, in reverse order
            odd_input_delays[i] =  -(((-self.N_TAPS)//2 + 1 + 2 * i) // self.N_SAMP_IN) + 2 # extra two stages to match C input with D input of DSP48
            stage_delay = 0
            self.SHIFTREGS_ODD[i] = Shiftreg(odd_input_delays[i])
            self.SHIFTREGS_EVEN[i] = [None for j in range(2*self.FILTER_DEPTH)]
            for j in reversed(range(self.FILTER_DEPTH)):
                even_input_delays[i,j,0] = -((2 * (i - j)) // self.N_SAMP_IN) + stage_delay
                even_input_delays[i,j,1] = -((2 * (i + j + 1) - self.N_TAPS - 1) // self.N_SAMP_IN) + stage_delay - 1 # need one less delay stage to match A input with D input of DSP48
                self.SHIFTREGS_EVEN[i][2*j] = Shiftreg(even_input_delays[i,j,0]) # D input
                self.SHIFTREGS_EVEN[i][2*j+1] = Shiftreg(even_input_delays[i,j,1]) # A input
                # account for delay from cascading multiple DSP48s:
                # note that this can be reduced by using a tree structure (which trades off using more DSP48s for fewer shift register stages)
                stage_delay += 2
                print(f'(i,j) = {i,j}, (x1,x2) = {((2 * (i - j)) % self.N_SAMP_IN, (2 * (i + j + 1) - self.N_TAPS - 1) % self.N_SAMP_IN)}')
        print(even_input_delays)
        print(odd_input_delays)

    def step(self, x_in):
        for i in range(self.N_SAMP_OUT):
            self.Y[i] = self.DSP48s[i][0].P
            for j in range(self.FILTER_DEPTH):
                # DSP48.step(A, D, C)
                x1 = x_in[(2 * (i - j)) % self.N_SAMP_IN]
                x2 = x_in[(2 * (i + j + 1) - self.N_TAPS - 1) % self.N_SAMP_IN]
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

    N_SAMP_ADC = 8 # samples per cycle from ADC

    # initialize FIR taps
    fir0 = FIR0()
    fir1 = FIR1()
    fir2 = FIR2()

    print(fir0.N_TAPS)

    hw_sim_fir0 = HW_HalfBandFIR(N_SAMP_ADC, FIR0())
    #hw_sim_fir1 = HW_HalfBandFIR(N_SAMP_ADC, FIR1())
    #hw_sim_fir2 = HW_HalfBandFIR(N_SAMP_ADC, FIR2())

    fir0_hw = np.zeros(N_in//2)
    #fir1_hw = np.zeros(N_in//2)
    #fir2_hw = np.zeros(N_in//2)

    for i in range(N_in//N_SAMP_ADC):
        fir0_hw[(N_SAMP_ADC//2)*i:(N_SAMP_ADC//2)*(i+1)] = hw_sim_fir0.Y
        hw_sim_fir0.step(x[N_SAMP_ADC*i:N_SAMP_ADC*(i+1)])
        #fir1_hw[(N_SAMP_ADC//2)*i:(N_SAMP_ADC//2)*(i+1)] = hw_sim_fir1.Y
        #hw_sim_fir1.step(x[N_SAMP_ADC*i:N_SAMP_ADC*(i+1)])
        #fir2_hw[(N_SAMP_ADC//2)*i:(N_SAMP_ADC//2)*(i+1)] = hw_sim_fir2.Y
        #hw_sim_fir2.step(x[N_SAMP_ADC*i:N_SAMP_ADC*(i+1)])

    #exit()

    # apply filter in software to signal
    fir0_sw = lfilter(fir0.TAPS, 1.0, x)[::2]
    fir1_sw = lfilter(fir1.TAPS, 1.0, x)[::2]
    fir2_sw = lfilter(fir2.TAPS, 1.0, x)[::2]
    fir0_sw_poly = lfilter(fir0.TAPS[0::2], 1.0, x[0::2]) + lfilter(fir0.TAPS[1::2] + [0], 1.0, np.concatenate(([0], x[1:-2:2])))
    fir1_sw_poly = lfilter(fir1.TAPS[0::2], 1.0, x[0::2]) + lfilter(fir1.TAPS[1::2] + [0], 1.0, np.concatenate(([0], x[1:-2:2])))
    fir2_sw_poly = lfilter(fir2.TAPS[0::2], 1.0, x[0::2]) + lfilter(fir2.TAPS[1::2] + [0], 1.0, np.concatenate(([0], x[1:-2:2])))

    # simulate hardware for filter

    # plot results
    fig, ax = plt.subplots(3,1,sharex=True)
    #fig, ax = plt.subplots(4,4)
    ax[0].plot(t_in, x, label='input')
    ax[1].plot(t_in[::2], fir0_sw_poly, label='fir0 software (polyphase) impl')
    ax[2].plot(t_in[::2], fir0_hw, label='fir0 hardware impl')
    #ax[0,3].plot(t_in[::2], fir0_sw - fir0_sw_poly, label='error')

    #ax[1,0].plot(t_in[::2], fir0_sw, label='fir0 software impl')
    #ax[1,1].plot(t_in[::2], fir0_sw_poly, label='fir0 software (polyphase) impl')
    #ax[1,2].plot(t_in[::2], fir0_hw, label='fir0 hardware impl')
    #ax[1,3].plot(t_in[::2], fir0_sw - fir0_sw_poly, label='error')

    #ax[2,0].plot(t_in[::2], fir1_sw, label='fir1 software impl')
    #ax[2,1].plot(t_in[::2], fir1_sw_poly, label='fir1 software (polyphase) impl')
    #ax[2,2].plot(t_in[::2], fir1_hw, label='fir1 hardware impl')
    #ax[2,3].plot(t_in[::2], fir1_sw - fir1_sw_poly, label='error')

    #ax[3,0].plot(t_in[::2], fir2_sw, label='fir2 software impl')
    #ax[3,1].plot(t_in[::2], fir2_sw_poly, label='fir2 software (polyphase) impl')
    #ax[3,2].plot(t_in[::2], fir2_hw, label='fir2 hardware impl')
    #ax[3,3].plot(t_in[::2], fir2_sw - fir2_sw_poly, label='error')

    plt.show()
