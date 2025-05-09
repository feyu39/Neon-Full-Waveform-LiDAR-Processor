<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor">
    <img src="images/NEON-NSF-Logo.png" alt="National Ecological Observatory Network Logo" width="80" height="80">
  </a>

<h3 align="center">National Ecological Observatory Network (NEON) Full Waveform LiDAR Processor and Biomass Analyzer</h3>

  <p align="center">
    This repository processes National Ecological Observatory Network (NEON) full-waveform LiDAR pulse waves (.pls and .wvs) data (DP1.30001.001). It also visualizes raw data, aboveground biomass regression functions, and k-means clustering functions. In-situ data for biomass regression is taken from the NEON Vegetation Structure (DP1.10098.001).
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

_[![Jupyter](https://files.seeklogo.com/download/8b155kpxr1ei2xma3jg63faljoyk7z/jupyter-vector-logo-seeklogo.zip)](https://jupyter.org/)
_[![Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)](https://www.python.org/)
_[![Scikit-learn](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)](https://scikit-learn.org/stable/)
_[![Numpy](https://github.com/numpy/numpy/blob/main/branding/logo/primary/numpylogo.png?raw=true)](https://numpy.org/) \*[![Pandas](https://pandas.pydata.org/static/img/pandas.svg)](https://pandas.pydata.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

Python 3.12 or later
Jupyter
Ipykernel
Numpy
Pandas
Matplotlib
scikit-learn
scipy
rioxarray

### Installation

1. Clone repository

```sh
git clone https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor.git
```

2. Create an Anaconda environment with environment.yml

- If you already have a kernel named “neon” replace the name part of the .yml file with whatever you want to name it, and use it instead of “neon” in subsequent instructions

```sh
conda env create -f environment.yml
```

3. Add environment as a Jupyter kernel

```sh
python -m ipykernel install –user –name neon –display-name “Python (neon)”
```

4. Activate conda environment and start jupyter lab

```sh
conda activate neon
jupyter lab
```

<!-- USAGE EXAMPLES -->

## Documentation/Usage

There are two main files: neon_processor_batch and neon_processor_full. Neon_processor_batch is a quicker way to analyze pulsewaves data than the full processor, but it does not filter based on geolocation data. Therefore, if you already know the pulsenumbers you want to analyze or want to test out functions first, use neon_processor_batch. However, if you want to filter out the pulsewaves file based on geolocation and analyze it, use neon_processor_full.

The general workflow of the code is to create a main pandas/excel table of waveform relative revelation data, waveform geolocation information, in situ biomass values, and k-means cluster assignments to get all the data in one place. Then, all analyses functions such as regression and k-means clustering are done by feeding in this table as input. See the **cumulative_waveform_analysis_main** function for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Neon_processor_batch functions

Important: Make sure to set the lidar_instrument_name and detection_threshold variables to the right sensors before running the code

**waveform_peak_detection(waveform, waveform_intensity_threshold)**
Quick and simple peak detection using derivatives and 2nd derivatives of the waveform.

Parameters:
waveform: _list (float)_

- List of intensity values of the waveform
  waveform*intensity_threshold: \_int*
- Minimum intensity threshold from which peaks are detected above

Returns:
return*location_count: \_int*

- Number of peaks  
  return*peak_location_list: \_list (int)*
- Array containing indices of waveform array where peaks are  
  return*location_list_x: \_list (float)*
- Same as return*peak_location_list, just in float format  
  return_intensity_list: \_list (float)*
- Intensity at each peak

**read_NEONAOP_pulsewaves_pulse_information_header_only(pls_file, lidar_instrument_name)**
Reads pulse metadata information from a NEON AOP PulseWaves (.pls) file without geolocation filtering, extracting variables necessary for future processing.

Parameters:
pls*file: \_str*

- Path to PulseWaves (.pls) file to read
  lidar*instrument_name : \_str*
- Name of the lidar instrument (e.g., 'Gemini', 'Galaxy2024', 'LMS-Q780')

Returns
A tuple containing the following elements:
instrument*name : \_str*

- Name of the instrument (e.g., 'Gemini', 'Galaxy', 'LMS-Q780')
  number*of_pulses : \_numpy.ndarray*
- Total number of pulses in the file
  xyz*anchor_array : \_numpy.ndarray*
- Array of anchor point coordinates (x,y,z) for each pulse
  dxdydz*array : \_numpy.ndarray*
- Array of direction vectors (dx,dy,dz) for each pulse
  xyz*first_array : \_numpy.ndarray*
- Array of first return coordinates (x,y,z) for each pulse
  xyz*last_array : \_numpy.ndarray*
- Array of last return coordinates (x,y,z) for each pulse
  offset*to_pulse_data : \_numpy.ndarray*
- Byte offset to the start of pulse data
  pulse*size : \_numpy.ndarray*
- Size of each pulse record in bytes
  T*scale_factor : \_numpy.ndarray*
- Scale factor for GPS time
  T*offset : \_numpy.ndarray*
- Offset for GPS time
  x*scale_factor : \_numpy.ndarray*
- Scale factor for x coordinates
  x*offset : \_numpy.ndarray*
- Offset for x coordinates
  y*scale_factor : \_numpy.ndarray*
- Scale factor for y coordinates
  y*offset : \_numpy.ndarray*
- Offset for y coordinates
  z*scale_factor : \_numpy.ndarray*
- Scale factor for z coordinates
  z*offset : \_numpy.ndarray*
- Offset for z coordinates
  sampling*record_pulse_descriptor_index_lookup_array : \_numpy.ndarray*
- Array mapping pulse descriptors to sampling records
  pulse*descriptor_optical_center_to_anchor_point_array : \_numpy.ndarray*
- Array of optical center to anchor point distances
  pulse*descriptor_number_of_extra_wave_bytes_array : \_numpy.ndarray*
- Array of extra waveform bytes per pulse
  pulse*descriptor_number_of_samplings_array : \_numpy.ndarray*
- Array of number of samplings per pulse
  sampling*record_bits_for_duration_from_anchor_array : \_numpy.ndarray*
- Array of bit counts for duration from anchor
  sampling*record_scale_for_duration_from_anchor_array : \_numpy.ndarray*
- Array of scale factors for duration from anchor
  sampling*record_offset_for_duration_from_anchor_array : \_numpy.ndarray*
- Array of offsets for duration from anchor
  sampling*record_bits_for_number_of_segments_array : \_numpy.ndarray*
- Array of bit counts for number of segments
  sampling*record_bits_for_number_of_samples_array : \_numpy.ndarray*
- Array of bit counts for number of samples
  sampling*record_number_of_segments_array : \_numpy.ndarray*
- Array of number of segments per sampling
  sampling*record_number_of_samples_array : \_numpy.ndarray*
- Array of number of samples per segment
  sampling*record_bits_per_sample_array : \_numpy.ndarray*
- Array of bits per sample

More details about the pulsewaves format can be found here at this <a href="https://github.com/PulseWaves/Specification/blob/master/specification.rst">link</a>

**read_NEONAOP_pulsewaves_waveform(readbin_pls_file,readbin_wvs_file,instrument_name,lidar_instrument_name,iPulse,offset_to_pulse_data,pulse_size,T_scale_factor,T_offset,x_scale_factor,x_offset,y_scale_factor,y_offset,z_scale_factor,z_offset,sampling_record_pulse_descriptor_index_lookup_array,pulse_descriptor_optical_center_to_anchor_point_array,pulse_descriptor_number_of_extra_wave_bytes_array,pulse_descriptor_number_of_samplings_array,sampling_record_bits_for_duration_from_anchor_array,sampling_record_scale_for_duration_from_anchor_array,sampling_record_offset_for_duration_from_anchor_array,sampling_record_bits_for_number_of_segments_array,sampling_record_bits_for_number_of_samples_array,sampling_record_number_of_segments_array,sampling_record_number_of_samples_array,sampling_record_bits_per_sample_array)**
Reads a single waveform (.wvs) file using metadata from Pulse (.pls) file, returning
waveform intensity values, easting, northing, elevation values, offset, and if waveform contains multiple segments

Parameters:
readbin*pls_file : \_str (path)*

- Open binary file object for the .pls file (PulseWaves).
  readbin*wvs_file : \_str (path)*
- Open binary file object for the .wvs file (Waveforms).
  instrument*name : \_str*
- Name of the instrument (e.g., 'Gemini', 'Galaxy').
  lidar*instrument_name : \_str*
- Full lidar system name (e.g., 'Galaxy2024', 'LMS-Q780').
  iPulse : _int_
- Index of the pulse to read
  offset*to_pulse_data : \_int*
- Byte offset to the beginning of pulse data in the .pls file.
  pulse*size : \_int*
- Size in bytes of each pulse record.
  T*scale_factor : \_float*
- Scale factor to convert GPS time to seconds.
  T*offset : \_float*
- Offset to add to scaled GPS time.
  x*scale_factor : \_float*
- Scale factor for x coordinates.
  x*offset : \_float*
- Offset for x coordinates.
  y*scale_factor : \_float*
- Scale factor for y coordinates.
  y*offset : \_float*
- Offset for y coordinates.
  z*scale_factor : \_float*
- Scale factor for z coordinates.
  z*offset : \_float*
- Offset for z coordinates.
  sampling*record_pulse_descriptor_index_lookup_array : \_numpy.ndarray*
- Array mapping pulse descriptors to sampling record indices.
  pulse*descriptor_optical_center_to_anchor_point_array : \_numpy.ndarray*
- Distance from optical center to anchor point for each pulse descriptor.
  pulse*descriptor_number_of_extra_wave_bytes_array : \_numpy.ndarray*
- Number of extra bytes associated with each waveform, by pulse descriptor.
  pulse*descriptor_number_of_samplings_array : \_numpy.ndarray*
- Number of waveform samplings for each pulse descriptor.
  sampling*record_bits_for_duration_from_anchor_array : \_numpy.ndarray*
- Bits used to represent the duration from the anchor for each sampling.
  sampling*record_scale_for_duration_from_anchor_array : \_numpy.ndarray*
- Scale factor for converting duration-from-anchor values.
  sampling*record_offset_for_duration_from_anchor_array : \_numpy.ndarray*
- Offset to apply to scaled duration-from-anchor values.
  sampling*record_bits_for_number_of_segments_array : \_numpy.ndarray*
- Bits used to encode number of segments in each waveform sampling.
  sampling*record_bits_for_number_of_samples_array : \_numpy.ndarray*
- Bits used to encode number of samples in each segment.
  sampling*record_number_of_segments_array : \_numpy.ndarray*
- Number of segments in the waveform if not explicitly encoded.
  sampling*record_number_of_samples_array : \_numpy.ndarray*
- Number of samples per segment if not explicitly encoded.
  sampling*record_bits_per_sample_array : \_numpy.ndarray*
- Bits used to store each sample in the waveform.

Returns:
neon*waveform_return_pulse : \_numpy.ndarray (float)*

- Array of intensity values for the waveform
  neon*waveform_x_axis : \_numpy.ndarray (float)*
- Array of easting values for the waveform
  neon*waveform_y_axis : \_numpy.ndarray (float)*
- Array of northing values for the waveform
  neon*waveform_z_axis : \_numpy.ndarray (float)*
- Array of elevation values for the waveform
  neon*waveform_offset : \_float*
- Smallest intensity value in the waveform

**normalize_cumulative_return_energy(waveform_intensity_x, waveform_elevation_y, plot)**
Normalize return energy (intensity) to a scale of 0.0 to 1.0 and get the relative elevation of the waveform at each intensity point by subtracting each elevation from the minimum waveform elevation point. Think about it as a “summary” of the raw waveform curve for all future analysis. Based on LVIS interpretation: <a href="https://lvis.gsfc.nasa.gov/Data/DataStructure.html">link</a>

Parameters:
waveform*intensity_x: \_1D array of floats*

- Raw waveform intensity/return energy values
  waveform*elevation_y: \_1D array of floats*
- Raw waveform elevation values in meters

Returns:
stacked*intensity_elevation: \_1D Numpy array of floats*

- Formatted as a 1D array: [Intensity1, Elevation1, Intensity2, Elevation2…] to feed into future scipy and scikit-learn functions

**interpolate_waveform_values(waveform, plot)**
Use scipy’s interpolate1d function to interpolate a curve based on existing normalized intensity and elevation values to ensure all normalized waveforms are the same input size. X intensity values range from 0 - 1.0 inclusive with 0.025 increments (41 values), and y elevation values are associated with each intensity increment.

Parameters:
waveform: _1D array of floats_
Stacked normalized cumulative return energy curve in the same format as [intensity1, elevation1…]
plot: _bool_
Plot the new interpolated normalized cumulative return energy curve if True

Returns:
interpolated*stacked_intensity_elevation: \_1D array of floats*
Interpolated cumulative signal intensity and interpolated relative elevation at each intensity stacked in a 1D array similar to previous formats of [intensity1, elevation1…]
outlier: _bool_
Set to true if final relative elevation value is above 150 meters (a processing error)

**calculate_silhouette_score(kmeans_fitted, X_train, k)**
The silhouette score is a metric of k-means clustering performance. From scikit-learn’s documentation: “​​The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.” This also helps determine the right k-number.
More information on scikit-learn’s guide

Parameters:
kmeans*fitted: \_scikit-learn object*
One scikit k-means trained classifier
X*train: \_1D array*
Training dataset that k-means was trained on
k: _int_
number of clusters

Returns:
Prints out the silhouette score of the model

**ccalculate_cumulative_signal_slopes(interpolated_waveforms, min_signal_level, max_signal_level)**
Calculate a linear slope for each normalized waveform between two intensity points. Sometimes used as a LiDAR metric in studies.

Parameters:
interpolated*waveforms: \_2D np.array*

- 2D np.array of interpolated cumulative waveforms
  min*signal_level: \_float*
- Minimum intensity point (factor 0.025) from which to determine slope from
  max*signal_level: \_float*
- Maximum intensity point (factor 0.025) from which to determine slope from

Returns:
Waveform*slopes: \_2D array of slopes for each index in the input*

**check_waveform_normality(interpolated_waveforms, waveform)**
Plot a QQ plot for intensity and elevation of each waveform or waveform slopes to see if distribution of waveforms is normal.

Parameters:
interpolated*waveforms: \_2D np.array*

- 2D np.array of interpolated cumulative waveforms or slopes (1D array)
  waveform: _bool_
- Whether the input is a waveform or slope

Returns:
QQ plot of waveform intensity, elevation, or slope

**perform_normal_waveform_outlier_analysis(waveform_arr)**
Calculate Z-score analysis and identify outliers 3 z-scores away

Parameters:
waveform*arr: \_1D array of floats*

- 1D array of all waveform slope values

Returns:
_1D array of indices_

- Array of indices which are higher than 3 z-scores away

**calculate_neighbors_distances(interpolated_waveforms, k)**
Calculate the mean distances to k-nearest neighbors for each waveform. Not used in analysis.

Parameters:
interpolated*waveforms: \_2D array*
2D array of interpolated cumulative waveforms
k: _int_
Number of clusters

<!-- ROADMAP -->

## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
  - [ ] Nested Feature

See the [open issues](https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feyu39/Neon-Full-Waveform-LiDAR-Processor" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Felix Yu - yu.felix39@gmail.com

Project Link: [https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor](https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- []()
- []()
- []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/feyu39/Neon-Full-Waveform-LiDAR-Processor.svg?style=for-the-badge
[contributors-url]: https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/feyu39/Neon-Full-Waveform-LiDAR-Processor.svg?style=for-the-badge
[forks-url]: https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/network/members
[stars-shield]: https://img.shields.io/github/stars/feyu39/Neon-Full-Waveform-LiDAR-Processor.svg?style=for-the-badge
[stars-url]: https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/stargazers
[issues-shield]: https://img.shields.io/github/issues/feyu39/Neon-Full-Waveform-LiDAR-Processor.svg?style=for-the-badge
[issues-url]: https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/issues
[license-shield]: https://img.shields.io/github/license/feyu39/Neon-Full-Waveform-LiDAR-Processor.svg?style=for-the-badge
[license-url]: https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/felixwyu
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
