"""This is the script that shows a simple SHG-FROG retrieval with pypret.

This is just a condensed version of the code explained in the Jupyter notebook
of the same name in this folder. Please refer to it for any questions.
"""
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypret
import json

# some general configuration options
PARSE = True
PLOT = True
SAVE = False

frg_path = "c:\\Users\\ritzk\\OneDrive\\Desktop\\SagarMeasurements\\FemtoFerb_FROG\\"
dataname = "FROG5_300fsrange_800steps_MANUALspecsettings_full_wav_range" # "FROG4_300fsrange_800steps_MANUALspecsettings" # 
fname = frg_path + dataname + ".tsv"

fname_fund = r"c:\Users\ritzk\OneDrive\Desktop\SagarMeasurements\Spectrum_1550nm_TOPTICA_1.csv"

# %% parse data only on demand
if PARSE:

    def load_frg(fname, isJSON = False, clipRaw=False, filtRaw=False, BW=False):
        """Load a single frog file in .frg format into a MeasurementTrace object.

        Assumes that the trace is equidistantly spaced in terms of the wavelength.
        """
        
        if not(isJSON):
            # load the whole file and store all values in a 1d-array
            with open(fname, "rt") as f:
                file = f.read().replace("\r\n", " ")
            raw = np.fromstring(file, sep=" ")

            M = int(raw[0])  # number of delay steps
            N = int(raw[1])  # number of wavelength/frequency points
                    
            # scale to basic units
            dtau = 1e-15 * raw[2]  # delay discretization
            dwl = 1e-9 * raw[3]  # wavelength discretization
            wl0 = 1e-9 * raw[4]  # central wavelength
            data = raw[5:].reshape(M, N)  # load in the data
            print(data.shape)

            # create the specific wavelength/delay grid
            wl = (np.arange(N) - (N // 2)) * dwl + wl0
            # m = (wl >= 700e-9) & (wl <= 850e-9)   # adjust to your SHG band
            # wl, data = wl[m], data[:, m]

            tau = (np.arange(M) - (M // 2)) * dtau

        else:
            json_path = Path(fname).with_suffix(".json")

            with open(json_path, "r", encoding="utf-8") as f:
                acquisition = json.load(f)

            wl = np.asarray(acquisition["Wavelengths"], dtype=float) * 1e-9
            tau = np.asarray(acquisition["PositionsInUnits"], dtype=float) * 1e-15
            data = np.asarray(acquisition["Data"], dtype=float).T

            M, N = data.shape
            assert data.shape == (tau.size, wl.size)
            assert np.all(np.diff(wl) > 0)
            assert np.all(np.diff(tau) > 0)

            dtau = np.mean(np.diff(tau))
            assert np.allclose(np.diff(tau), dtau, rtol=1e-5, atol=0)


        if filtRaw:
            # remove the 196 THz interferometric fringes, smooth roll-off
            f = np.fft.rfftfreq(M, dtau)
            F = np.fft.rfft(data, axis=0)
            F *= np.exp(-(f / 145e12)**4)[:, None]
            data = np.fft.irfft(F, n=M, axis=0)

        if clipRaw:
            if BW:
                            
                # f = np.fft.rfftfreq(M, dtau)
                # F = np.fft.rfft(data, axis=0)
                # F[f > 180e12, :] = 0.0 #60e12
                # data = np.fft.irfft(F, n=M, axis=0)
                # # data = 0.5 * (data + data[::-1, :])
                band = (wl >= 700e-9) & (wl <= 860e-9)
                data -= np.median(data[:, ~band], axis=1, keepdims=True)  # per-delay dark offset
                data[:, ~band] = 0.0                                      # zero out-of-band noise
                data[data < 0] = 0.0  
            else:
                #    per-delay dark subtraction
                band = (wl >= 700e-9) & (wl <= 860e-9)
                data -= np.median(data[:, ~band], axis=1, keepdims=True)

            # soft taper edges
            data *= np.exp(-((wl - 780e-9) / 90e-9)**8)[None, :]
            

        # convert data (M, N) array into list of M Spectrum instances
        spectra = []
        for intensity in data:
            spectra.append(pypret.Spectrum(wl, intensity))
        # create MeasurementTrace with correct method specifier
        trace = pypret.MeasurementTrace(tau, spectra, method="frog", process="shg")
        return trace

    def load_fundamental(fname):
        """Load a fundamental spectrum from the csv files."""
        # skiprows simply ignores the header
        df = pd.read_csv(fname, )
        wavelength, intensity = df['Wavelength [m]'].to_numpy(), df['Amp [a.u.]'].to_numpy()
        print("WL:", wavelength[1:10])
        print("\n\nIntensity:", intensity[1:10])
        wl = wavelength #already in meters, don't need to * 1e-9
        return pypret.Spectrum(wl, intensity)

    measurement = load_frg(fname, isJSON=True, clipRaw=True, filtRaw=True, BW=False ) #True False


    # measurement.parameter -= 1.96e-15  # FROG5: shift estimated center to zero

    # edge = np.abs(measurement.parameter) > 130e-15
    # background = np.median(measurement.data[edge], axis=0)
    # measurement.add(-background)
    
    fundamental = load_fundamental(fname_fund)

    # use the in-built file compression to save space, not necessary
    if SAVE:
        pypret.save((fundamental, measurement), "frog_" + dataname + ".h5.7z", archive=True)
else:
    # load previously parsed data
    fundamental, measurement = pypret.load("frog_" + dataname + ".h5.7z")

if PLOT:
    pypret.graphics.TracePlot(
        measurement, marginals=True, suptitle="raw trace", cmap="nipy_spectral"
    )

# %% process the FROG trace
# measurement.limit_parameter((-9.5e-12, None))  # remove erroneous data


#does same thing as clip in load_frg, but can be used to remove dark signal in a specific wavelength range'
# measurement.remove_dark_signal((300e-9, 700e-9), (880e-9, 1050e-9))
# 
# pulse = measurement.grid()  # interpolate on simulation grid

ft = pypret.FourierTransform(1024, dt=5e-15)
pulse = pypret.Pulse(ft, 1565e-9)
grid_model = pypret.PNPS(pulse, "frog", "shg")
measurement.interpolate(grid_model.process_wl)

if PLOT:
    pypret.graphics.TracePlot(
        measurement, marginals=True, cmap="nipy_spectral", title="interpolated"
    )

# %% process the fundamental spectrum
if PLOT:
    fundamental.show(title="raw fundamental")

# remove remaining CW
# fundamental.limit((None, 1.45e-6), (1.65e-6, None))
# fundamental.limit((None, 1.55e-6), (1.555e-6, None))

if PLOT:
    fundamental.show(title="CW removed")


# %% calculate the transform-limited pulse duration
spectrum = fundamental.copy()
# spectrum.smooth(0.5e-9)
# spectrum.bandpass((1.4e-6, 1.7e-6), dx=0.3e-9)
tl_pulse = spectrum.create_pulse(pulse.copy())
print("TL FWHM: %.1f fs" % (tl_pulse.fwhm(dt=tl_pulse.dt / 100) * 1e15))
if PLOT:
    spectrum.show(title="smoothed and apodized")
    pypret.graphics.PulsePlot(tl_pulse, oversampling=4)


# %% define the PNPS instance and the objective
pnps = pypret.PNPS(pulse, "frog", "shg")

# calculate the trace of the reference spectrum
objective = pypret.LeastSquaresObjective(pnps, measurement)

# %% compare the trace to the trace of the transform-limited pulse
if PLOT:
    trace = objective.trace(tl_pulse.spectrum)
    pypret.graphics.CompareTraceMeshPlot(
        measurement, trace, suptitle="comparison with TL pulse", marginals=True
    )


# %% perform the retrieval
fname_out = "retrieved_frog_" + dataname + ".hdf5"
# if not Path(fname_out).exists():
res = pypret.retrieve(
    pnps,
    measurement,
    repetitions=16,
    verbose=True,
    maxiter=300,
    measured_spectrum=fundamental,
    guess=tl_pulse.spectrum,
) #pypret.retrieve(pnps, measurement, repetitions=4, maxiter=100, verbose=True, ...)

if SAVE:
    res.save(fname_out)

# else:
#     res = pypret.load(fname_out)


## obtain and plot the mean result
# mean = res.mean(mode="no_outliers")
# rpulse = mean.pulse
# mean.show()
# print(
#     "Retrieved pulse duration (FWHM): %.1f fs"
#     % (rpulse.fwhm(dt=tl_pulse.dt / 100) * 1e15)
# )


# # Option for the pulse with lowest trace error
mean = res.best
rpulse = mean.pulse
mean.show()
print(
    "Retrieved pulse duration (FWHM): %.1f fs"
    % (rpulse.fwhm(dt=tl_pulse.dt / 100) * 1e15)
)

print("Best trace error:", mean.trace_error)

measured = mean.measured_trace.data
retrieved = mean.retrieved_trace.data
print("Relative residual:",
      np.linalg.norm(measured - retrieved) / np.linalg.norm(measured))

# %%
if SAVE:
    import h5py
    f = h5py.File(fname_out, "r")
    print(f.keys())  # Prints root keys/groups
    print(f)



# %%

intensity = rpulse.intensity
time_from_peak = rpulse.t - rpulse.t[np.argmax(intensity)]

outside = np.abs(time_from_peak) > 150e-15
fraction = intensity[outside].sum() / intensity.sum()

print(f"Energy outside ±150 fs of the peak: {100 * fraction:.2f}%")

 
# %%
# start = res.best

# refine_objective = pypret.LeastSquaresObjective(
#     pypret.PNPS(start.pulse, "frog", "shg"),
#     start.measured_trace,
# )

# solver = pypret.Retriever("copra", maxiter=600, verbose=True)
# solver.retrieve(refine_objective, start.retrieved_spectrum.copy())
# refined = solver.result(measured_spectrum=fundamental)

# p = refined.pulse
# I = p.intensity
# t = p.t - p.t[np.argmax(I)]

# print("Before trace error:", start.trace_error)
# print("After trace error:", refined.trace_error)
# print("FWHM [fs]:", p.fwhm(dt=p.dt / 100) * 1e15)
# print("Energy outside ±150 fs [%]:",
#       100 * I[np.abs(t) > 150e-15].sum() / I.sum())

# refined.show(trace_plot=False)





# # %%

# check_pulse = res.best.pulse

# check_measured_tl = fundamental.create_pulse(check_pulse.copy())

# check_retrieved_tl = check_pulse.copy()
# check_retrieved_tl.spectrum = np.abs(check_pulse.spectrum).astype(complex)

# for label, candidate in [
#     ("TL from measured spectrum", check_measured_tl),
#     ("TL from retrieved spectrum", check_retrieved_tl),
#     ("Retrieved pulse", check_pulse),
# ]:
#     width_fs = candidate.fwhm(dt=candidate.dt / 100) * 1e15
#     print(f"{label}: {width_fs:.2f} fs")


# %%

probe = res.best.pulse
amplitude = np.abs(probe.spectrum)

inside = (
    (probe.wl >= fundamental.wavelength.min())
    & (probe.wl <= fundamental.wavelength.max())
)

full_tl = probe.copy()
full_tl.spectrum = amplitude.astype(complex)

limited_tl = probe.copy()
limited_tl.spectrum = (amplitude * inside).astype(complex)

outside_energy = 100 * np.sum(amplitude[~inside]**2) / np.sum(amplitude**2)

print(f"Spectral energy outside measured range: {outside_energy:.3f}%")
print(f"Full retrieved TL: {full_tl.fwhm(dt=probe.dt/100)*1e15:.2f} fs")
print(f"Restricted retrieved TL: {limited_tl.fwhm(dt=probe.dt/100)*1e15:.2f} fs")

# %%

p_check = res.best.pulse
m_check = fundamental.create_pulse(p_check.copy())

order = np.argsort(p_check.wl)
wl_nm = p_check.wl[order] * 1e9

retrieved_I = p_check.spectral_intensity_wl[order]
measured_I = m_check.spectral_intensity_wl[order]

plt.figure(figsize=(8, 4))
plt.semilogy(wl_nm, np.maximum(retrieved_I / retrieved_I.max(), 1e-8),
             label="Retrieved")
plt.semilogy(wl_nm, np.maximum(measured_I / measured_I.max(), 1e-8),
             label="Measured")
plt.xlim(fundamental.wavelength.min() * 1e9,
         fundamental.wavelength.max() * 1e9)
plt.ylim(1e-5, 2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral intensity / own peak")
plt.grid(True)
plt.legend()
plt.show()
# %%
