from deltasigma import *
import numpy as np
from matplotlib.pyplot import *
import warnings
from scipy.signal import ss2zpk
warnings.filterwarnings('ignore')
order = 2
osr = 256 # oversampling ratio
nlev = 2 # adc quantization level - 2 bits since single comparator is used
f0 = 0.0 # center frequency 0
Hinf = 2 # maximum out-of-band NTF gain
tdac = [0, 1] # pulse timing (NRZ)
form = 'FB' # cascade of integrators, feedback
M = nlev - 1 # quantizer step normalized to +/- 1
plotsize = (6,6)

print(f"{order}-order CT-DSM")
print("Doing NTF Synthesis...")
ntf0 = synthesizeNTF(order, osr, 2, Hinf, f0) # returns [(z1, p1), ...] for NTF 
print("Done.")
print(f"Synthesized a {order}-order NTF, with roots: ")
print("Zeroes: \t\t\t Poles:")
for z, p in zip(ntf0[0], ntf0[1]):
    print(f"({np.real(z):.8f},{np.imag(z):.8f}j), ({np.real(p):.8f}, {np.imag(p):.8f}j)")

print("Generating NTF pole-zero plot...")
DocumentNTF(ntf0, osr, f0) # NTF pole-zero plot and low-pass NTF frequency response
figure(figsize=plotsize)
PlotExampleSpectrum(ntf0, M, osr, f0) # Time-domain simulation with input cosine signal. Try invoking simulateDSM by itself

if nlev == 2:
    snr_pred, amp_pred, k0, k1, se = predictSNR(ntf0, osr)
snr, amp = simulateSNR(ntf0, osr, None, f0, nlev) # SNQR simulation

figure(figsize=plotsize)
if nlev == 2:
    plot(amp_pred, snr_pred, '-', label='Predicted')
plot(amp, snr,'o-.g', label='simulated')
xlabel('Input Level (dBFS)')
ylabel('SQNR (dB)')
peak_snr, peak_amp = peakSNR(snr, amp)
msg = 'peak SQNR = %4.1fdB  \n@ amp dd= %4.1fdB  ' % (peak_snr, peak_amp)
text(peak_amp-10, peak_snr, msg, horizontalalignment='right', verticalalignment='center');
msg = 'OSR = %d ' % osr
text(-2, 5, msg, horizontalalignment='right');
figureMagic([-80, 0], 10, None, [0, 120], 10, None, [12, 6], 'Time-Domain Simulations')
legend(loc=2);

print("Mapping to continuous time...")
ABCDc, tdac2 = realizeNTF_ct(ntf0, form, tdac)

# tdac2 returns  

Ac, Bc, Cc, Dc = partitionABCD(ABCDc)
sys_c = []
for i in range(Bc.shape[1]):
    sys_c.append(ss2zpk(Ac, Bc, Cc, Dc, input=i))

print("done.")
print(f"ABCD matrix: {ABCDc}")
print(f"DAC timing (tdac2): {tdac2}")


figure(figsize=plotsize)
n_imp = 10
y = -impL1(ntf0, n_imp)
lollipop(np.arange(n_imp + 1), y)
grid(True)
dt = 1./16
tppulse = np.vstack((np.zeros((1, 2)), tdac2[1:, :])).tolist()
yy = -pulse(sys_c, tppulse, dt, n_imp).squeeze()
t = np.linspace(0, n_imp + dt, int(10/dt + 1))
plot(t, yy, 'g', label='continuous-time')
legend()
title('Loop filter pulse/impulse responses (negated)');

show()
