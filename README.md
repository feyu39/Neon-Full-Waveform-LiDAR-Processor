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
    <img src="images/NEON-NSF-Logo.png" alt="National Ecological Observatory Network Logo" width="200" height="200">
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

<p align="left">
  <a href="https://jupyter.org/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/207px-Jupyter_logo.svg.png" width="150"/>
  </a>
  <a href="https://www.python.org/">
    <img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width="150"/>
  </a>
  <a href="https://scikit-learn.org/stable/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="150"/>
  </a>
  <a href="https://numpy.org/">
    <img src="https://github.com/numpy/numpy/blob/main/branding/logo/primary/numpylogo.png?raw=true" width="150"/>
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://pandas.pydata.org/static/img/pandas.svg" width="150"/>
  </a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

- Python 3.12 or later
- Jupyter
- Ipykernel
- Numpy
- Pandas
- Matplotlib
- scikit-learn
- scipy
- rioxarray

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

### neon_processor_batch functions

Important: Make sure to set the lidar_instrument_name and detection_threshold variables to the right sensors before running the code

**waveform_peak_detection(waveform, waveform_intensity_threshold)**
- Quick and simple peak detection using derivatives and 2nd derivatives of the waveform.

Parameters:
waveform: _list (float)_
- List of intensity values of the waveform
  waveform_intensity_threshold: _int_
- Minimum intensity threshold from which peaks are detected above

Returns:
return_location_count: _int_
- Number of peaks  
return_peak_location_list: _list(int)_
- Array containing indices of waveform array where peaks are  
return_location_list_x: _list (float)_
- Same as return_peak_location_list, just in float format  
return_intensity_list: _list (float)_
- Intensity at each peak

**read_NEONAOP_pulsewaves_pulse_information_header_only(pls_file, lidar_instrument_name)**
- Reads pulse metadata information from a NEON AOP PulseWaves (.pls) file without geolocation filtering, extracting variables necessary for future processing.

Parameters:
pls_file: *str*
- Path to PulseWaves (.pls) file to read
  lidar_instrument_name: *str*
- Name of the lidar instrument (e.g., 'Gemini', 'Galaxy2024', 'LMS-Q780')

Returns:
A tuple containing the following elements:
instrument_name: *str*
- Name of the instrument (e.g., 'Gemini', 'Galaxy', 'LMS-Q780')
number_of_pulses: *numpy.ndarray*
- Total number of pulses in the file
xyz_anchor_array: *numpy.ndarray*
- Array of anchor point coordinates (x,y,z) for each pulse
dxdydz_array: *numpy.ndarray*
- Array of direction vectors (dx,dy,dz) for each pulse
xyz_first_array: *numpy.ndarray*
- Array of first return coordinates (x,y,z) for each pulse
xyz_last_array: *numpy.ndarray*
- Array of last return coordinates (x,y,z) for each pulse
offset_to_pulse_data: *numpy.ndarray*
- Byte offset to the start of pulse data
pulse_size: *numpy.ndarray*
- Size of each pulse record in bytes
T_scale_factor: *numpy.ndarray*
- Scale factor for GPS time
T_offset: *numpy.ndarray*
- Offset for GPS time
x_scale_factor: *numpy.ndarray*
- Scale factor for x coordinates
x_offset: *numpy.ndarray*
- Offset for x coordinates
y_scale_factor: *numpy.ndarray*
- Scale factor for y coordinates
y_offset: *numpy.ndarray*
- Offset for y coordinates
z_scale_factor: *numpy.ndarray*
- Scale factor for z coordinates
z_offset: *numpy.ndarray*
- Offset for z coordinates
sampling_record_pulse_descriptor_index_lookup_array: *numpy.ndarray*
- Array mapping pulse descriptors to sampling records
pulse_descriptor_optical_center_to_anchor_point_array: *numpy.ndarray*
- Array of optical center to anchor point distances
pulse_descriptor_number_of_extra_wave_bytes_array: *numpy.ndarray*
- Array of extra waveform bytes per pulse
pulse_descriptor_number_of_samplings_array: *numpy.ndarray*
- Array of number of samplings per pulse
sampling_record_bits_for_duration_from_anchor_array: *numpy.ndarray*
- Array of bit counts for duration from anchor
sampling_record_scale_for_duration_from_anchor_array: *numpy.ndarray*
- Array of scale factors for duration from anchor
sampling_record_offset_for_duration_from_anchor_array: *numpy.ndarray*
- Array of offsets for duration from anchor
sampling_record_bits_for_number_of_segments_array: *numpy.ndarray*
- Array of bit counts for number of segments
sampling_record_bits_for_number_of_samples_array: *numpy.ndarray*
- Array of bit counts for number of samples
sampling_record_number_of_segments_array: *numpy.ndarray*
- Array of number of segments per sampling
sampling_record_number_of_samples_array: *numpy.ndarray*
- Array of number of samples per segment
sampling_record_bits_per_sample_array: *numpy.ndarray*
- Array of bits per sample

More details about the pulsewaves format can be found here at this <a href="https://github.com/PulseWaves/Specification/blob/master/specification.rst">link</a>

**read_NEONAOP_pulsewaves_waveform(readbin_pls_file,readbin_wvs_file,instrument_name,lidar_instrument_name,iPulse,offset_to_pulse_data,pulse_size,T_scale_factor,T_offset,x_scale_factor,x_offset,y_scale_factor,y_offset,z_scale_factor,z_offset,sampling_record_pulse_descriptor_index_lookup_array,pulse_descriptor_optical_center_to_anchor_point_array,pulse_descriptor_number_of_extra_wave_bytes_array,pulse_descriptor_number_of_samplings_array,sampling_record_bits_for_duration_from_anchor_array,sampling_record_scale_for_duration_from_anchor_array,sampling_record_offset_for_duration_from_anchor_array,sampling_record_bits_for_number_of_segments_array,sampling_record_bits_for_number_of_samples_array,sampling_record_number_of_segments_array,sampling_record_number_of_samples_array,sampling_record_bits_per_sample_array)**
- Reads a single waveform (.wvs) file using metadata from Pulse (.pls) file, returning
waveform intensity values, easting, northing, elevation values, offset, and if waveform contains multiple segments

Parameters:
readbin_pls_file: *str (path)*
- Open binary file object for the .pls file (PulseWaves).
readbin_wvs_file: *str (path)*
- Open binary file object for the .wvs file (Waveforms).
instrument_name: *str*
- Name of the instrument (e.g., 'Gemini', 'Galaxy').
lidar_instrument_name: *str*
- Full lidar system name (e.g., 'Galaxy2024', 'LMS-Q780').
iPulse: _int_
- Index of the pulse to read
offset_to_pulse_data: _int_
- Byte offset to the beginning of pulse data in the .pls file.
pulse_size: _int_
- Size in bytes of each pulse record.
T_scale_factor: _float_
- Scale factor to convert GPS time to seconds.
T_offset: _float_
- Offset to add to scaled GPS time.
x_scale_factor: _float_
- Scale factor for x coordinates.
x_offset: _float_
- Offset for x coordinates.
y_scale_factor: _float_
- Scale factor for y coordinates.
y_offset: _float_
- Offset for y coordinates.
z_scale_factor: _float_
- Scale factor for z coordinates.
z_offset: _float_
- Offset for z coordinates.
sampling_record_pulse_descriptor_index_lookup_array: _numpy.ndarray_
- Array mapping pulse descriptors to sampling record indices.
pulse_descriptor_optical_center_to_anchor_point_array: _numpy.ndarray_
- Distance from optical center to anchor point for each pulse descriptor.
pulse_descriptor_number_of_extra_wave_bytes_array: _numpy.ndarray_
- Number of extra bytes associated with each waveform, by pulse descriptor.
pulse_descriptor_number_of_samplings_array: _numpy.ndarray_
- Number of waveform samplings for each pulse descriptor.
sampling_record_bits_for_duration_from_anchor_array: _numpy.ndarray_  
- Bits used to represent the duration from the anchor for each sampling.
sampling_record_scale_for_duration_from_anchor_array: _numpy.ndarray_
- Scale factor for converting duration-from-anchor values.
sampling_record_offset_for_duration_from_anchor_array: _numpy.ndarray_
- Offset to apply to scaled duration-from-anchor values.
sampling_record_bits_for_number_of_segments_array: _numpy.ndarray_
- Bits used to encode number of segments in each waveform sampling.
sampling_record_bits_for_number_of_samples_array: _numpy.ndarray_
- Bits used to encode number of samples in each segment.
sampling_record_number_of_segments_array: _numpy.ndarray_
- Number of segments in the waveform if not explicitly encoded.
sampling_record_number_of_samples_array: _numpy.ndarray_
- Number of samples per segment if not explicitly encoded.
sampling_record_bits_per_sample_array: _numpy.ndarray_
- Bits used to store each sample in the waveform.

Returns:
neon_waveform_return_pulse: _numpy.ndarray (float)_
- Array of intensity values for the waveform
neon_waveform_x_axis: _numpy.ndarray (float)_
- Array of easting values for the waveform
neon_waveform_y_axis: _numpy.ndarray (float)_
- Array of northing values for the waveform
neon_waveform_z_axis: _numpy.ndarray (float)_
- Array of elevation values for the waveform
neon_waveform_offset: _float_
- Smallest intensity value in the waveform

**normalize_cumulative_return_energy(waveform_intensity_x, waveform_elevation_y, plot)**
- Normalize return energy (intensity) to a scale of 0.0 to 1.0 and get the relative elevation of the waveform at each intensity point by subtracting each elevation from the minimum waveform elevation point. Think about it as a “summary” of the raw waveform curve for all future analysis. Based on LVIS interpretation: <a href="https://lvis.gsfc.nasa.gov/Data/DataStructure.html">link</a>

Parameters:
waveform_intensity_x: *1D array of floats*

- Raw waveform intensity/return energy values
waveform_elevation_y: *1D array of floats*
- Raw waveform elevation values in meters

Returns:
stacked_intensity_elevation: *1D array of floats*
- Formatted as a 1D array: [Intensity1, Elevation1, Intensity2, Elevation2…] to feed into future scipy and scikit-learn functions

**interpolate_waveform_values(waveform, plot)**
- Use scipy’s interpolate1d function to interpolate a curve based on existing normalized intensity and elevation values to ensure all normalized waveforms are the same input size. X intensity values range from 0 - 1.0 inclusive with 0.025 increments (41 values), and y elevation values are associated with each intensity increment.

Parameters:
waveform: *1D array of floats*
- Stacked normalized cumulative return energy curve in the same format as [intensity1, elevation1…]
plot: *bool*
- Plot the new interpolated normalized cumulative return energy curve if True

Returns:
interpolated_stacked_intensity_elevation: *1D array of floats*
- Interpolated cumulative signal intensity and interpolated relative elevation at each intensity stacked in a 1D array similar to previous formats of [intensity1, elevation1…]
outlier: *bool*
- Set to true if final relative elevation value is above 150 meters (a processing error)

**calculate_silhouette_score(kmeans_fitted, X_train, k)**
- The silhouette score is a metric of k-means clustering performance. From scikit-learn’s documentation: “​​The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.” This also helps determine the right k-number.
- More information on scikit-learn’s guide

Parameters:
kmeans_fitted: *scikit-learn object*
- One scikit k-means trained classifier
X_train: *1D array*
- Training dataset that k-means was trained on
k: *int*
- number of clusters

Returns:
Prints out the silhouette score of the model

**calculate_cumulative_signal_slopes(interpolated_waveforms, min_signal_level, max_signal_level)**
- Calculate a linear slope for each normalized waveform between two intensity points. Sometimes used as a LiDAR metric in studies.

Parameters:
interpolated_waveforms: *2D np.array*
- 2D np.array of interpolated cumulative waveforms
min_signal_level: *float*
- Minimum intensity point (factor 0.025) from which to determine slope from
max_signal_level: *float*
- Maximum intensity point (factor 0.025) from which to determine slope from

Returns:
waveform_slopes: *2D array of slopes for each index in the input*

**check_waveform_normality(interpolated_waveforms, waveform)**
- Plot a QQ plot for intensity and elevation of each waveform or waveform slopes to see if distribution of waveforms is normal.

Parameters:
interpolated_waveforms: *2D np.array*
- 2D np.array of interpolated cumulative waveforms or slopes (1D array)
waveform: *bool*
- Whether the input is a waveform or slope

Returns:
QQ plot of waveform intensity, elevation, or slope

**perform_normal_waveform_outlier_analysis(waveform_arr)**
- Calculate Z-score analysis and identify outliers 3 z-scores away

Parameters:
waveform_arr: *_1D array of floats_*
- 1D array of all waveform slope values

Returns:
1D array of indices
- Array of indices which are higher than 3 z-scores away

**calculate_neighbors_distances(interpolated_waveforms, k)**
- Calculate the mean distances to k-nearest neighbors for each waveform. Not used in analysis.

Parameters:
interpolated_waveforms: *2D array*
- 2D array of interpolated cumulative waveforms
k: *int*
- Number of clusters

**train_k_means_cluster_cumulative_returns(interpolated_waveforms_train, k_list, evaluate_elbow)**
- Train a scikit-learn k-means cluster object for every k in the k-list

Parameters:
interpolated_waveforms_train: *2D Numpy array of floats*
- 2D numpy array of interpolated waveforms in the same format as other functions [intensity1, elevation1…]
k-list: *Array of ints for k-number of clusters*
evaluate_elbow: *bool*
- If set to True, plot the elbow method heuristic, the log sum of squared distance (loss function) at each k. The k at which an “elbow” appears should be chosen as the number of clusters. 

Returns:
k_means_classifiers: 1D array of k-means trained classifiers for each k in the k-list

**test_k_means_cluster_cumulative_returns(k_means_classifiers, X_test, k_list, plot_raw, evaluate_metrics)**
- Get the cluster centers and assign each waveform to a cluster based for each trained k-means classifier.

Parameters:
k_means_classifiers: *1D array of scikit-learn k-means classifier objects*
X_test: *1D Numpy array of waveforms in intensity1, elevation1… format*
k_list: *1D array of number of clusters associated with each k-means classifier object*
plot_raw: *bool*
- Set true to plot the X_test waveform data
evaluate_metrics: *bool*
- Get the silhouette and davies bouldin score of each k-means cluster object if set to true

Returns:
Cluster_centers: *3D np.array*
- A list of 2D Numpy arrays of cluster centers, where each 2D numpy array is a of size (number_of_clusters, waveform_array) for each k-means classifier (each cluster has its own waveform representing the center)
test_cluster_assignments: *2D array*
- A list of 1D arrays containing the cluster number label (integer) that each waveform belongs to for each k_means classifier object. For example, if there are two classifier objects, then this variable would contain two 1D arrays with the size of the number of samples in X_test

*get_apparent_individual_biomass_information(apparent_individual_table, mapping_tagging_table, resolution, debug)*
- Get the geolocation information and NEON individual ID of each biomass measurement that meets the resolution requirements. Based on NEON Vegetation Structure data product

Parameters:
apparent_individual_table: *str*
- Folder directory link to the vst_apparentindividual table for a specific date taken from NEON vegetation structure data. This table contains in situ data for diameter breast height, canopy height, and rough geolocation data
mapping_tagging_table: *str*
- Folder directory link to the vst_mapping and tagging table for a specific date taken from NEON vegetation structure data. This table contains precise geolocation information and species classification data
resolution: *int*
- Int for the max size of each biomass measurement. Coarser resolutions will include more points but less precision
debug: *bool*
- Print out location and ID information if True

Returns:
northing_arr: *1D np.array of floats*
- Array of northing values corresponding to the biomass measurements
easting_arr: *1D np.array of floats*
- Array of easting values corresponding to the biomass measurements
individual_id_arr: *1D np.array of strings*
- Array of NEON individual IDs corresponding to the biomass measurements
total_data_count: *int*
- Total number of biomass measurements that meet the resolution requirements

**calculate_indv_tree_biomass(genus, family, species)**
- Lookup table for tree biomass regression coefficients based on species and corresponding specific gravity taken from the US Forest Service Chojnacky paper.

Parameters:
Family: *str*
- Plant family
Genus: *str*
- Plant genus
Species: *str*
- Plant species, with the first letter lowercase

Returns:
B0: *float*
- y-intercept coefficient of tree biomass
B1: *float*
- Slope coefficient of tree biomass, multiply by np.log(dbh)

**calculate_total_tree_biomass(northing_area, easting_area, resolution, mapping_tagging_table, apparent_individual_table)**
- Calculate total biomass value in the waveform northing and easting point and any surrounding area based on the resolution.

Parameters:
northing_area: *float*
- Northing coordinate of the waveform
easting_area: *float*
- Easting coordinate of the waveform
resolution: *int*
- Resolution to get the surrounding biomass area
apparent_individual_table: *str*
- Folder directory link to vst_apparentindividual table

Returns:
total_biomass: *float*
- Float with the total biomass in kg of the area

find_associated_biomass_waveform_group(waveform_x_axis, waveform_y_axis, grouped_biomass, shot_number)
- See if the geolocation of a waveform is located among a group of biomass geolocation values. Used to locate if a waveform has in situ data associated with it. Assumes that general range of biomass geolocation values is known.

Parameters
waveform_x_axis: *1D Array*
- Waveform easting values
waveform_y_axis: *1D Array*
- Waveform northing values
grouped_biomass: *2D Array*
- Biomass geolocation range in the form of [minimum easting, maximum easting, minimum northing, maximum northing] for each biomass group
shot_number: *int*
- Pulsewave number of waveform

Returns:
Biomass_found: *bool*
- Bool that is set true if in situ biomass value is found for the pulsewave
Group: *int*
- The group that the associated biomass number belongs to, based on its index

**find_associated_biomass_waveform_pulse_number(biomass_northing_arr, biomass_easting_arr, waveform_x_axis, waveform_y_axis, debug)**
- Get the precise mean of the northing and easting values of a waveform if it is associated with a biomass value.

Parameters:
Biomass_northing_arr: *1D array of floats*
- Float array of all in situ biomass northing values returned from get_apparent_individual_biomass_information
Biomass_easting_arr: *1D array of floats*
- Float array of all in situ biomass easting values returned from get_apparent_individual_biomass_information
Waveform_x_axis: *1D array of floats*
- Float array of easting geolocation values of the waveform 
waveform_y_axis: *1D array of floats*
- Float array of northing geolocation values of the waveform 
resolution: *int*
- Area to search around biomass geolocation values to see if waveform geolocation falls into the biomass area. Should match calculate_tree_biomass resolution value for most accurate results
debug: *bool*
- Print out geolocation and number of biomass values associated with the waveform if True

Returns:
biomass_found: *bool*
- Whether or not at least one in-situ biomass value was found for the waveform
waveform_easting: *float*
- Mean of easting values for the waveform
waveform_northing: *float*
- Mean of northing values for the waveform

**find_biomass_tree_heights(northing, easting, resolution, mapping_tagging_table, apparent_individual_table)**
- Get in-situ tree heights for each waveform based on geolocation information.

Parameters:
northing: *float*
- Float of waveform northing value
easting: *float*
- Float of waveform easting value
resolution: *int*
- Integer of resolution to search area around waveform northing and easting. Should match get_apparent_individual_biomass_information resolution
mapping_tagging_table: *str*
- Folder directory link to the vst_mappingandtagging table from the NEON vegetation structure dataset
apparent_individual_table: *str*
- Folder directory link to the vst_apparentindividual table from the NEON vegetation structure dataset

Returns
Height: Float of the total in situ height of the tree in meters

**add_waveform_data_row_with_biomass(waveform_file_name, interp_waveform, waveform_x_axis, waveform_y_axis, biomass_value, cluster_number, shot_number, biomass_group_number)**
- Helper function to add a row to the main biomass-waveform table given extra data using a pandas Dataframe.
- Each row of the main biomass-waveform table contains these indexes in order: FileName, Pulsenumber, Biomass, Easting, Northing, Group, Cluster

Parameters:
Waveform_file_name: *str* 
- Contains the file postfix deliminating the region(such as L022) 
Interp_waveform: *1D array*
- Waveform data with intensity1.., elevation1…
Waveform_x_axis: *1D array of floats*
- Waveform easting values
Waveform_y_axis: *1D array of floats*
- Waveform northing values
Biomass_value: *float*
- In-situ biomass value in kg
Cluster_number: *int*
- Cluster that waveform is assigned to
Shot_number: *int*
- Pulsenumber of the waveform
Biomass_group_number: *int*
- Group number that corresponds to the return value in find_associated_biomass_waveform_group

Returns:
Waveform_biomass_df: *pandas dataframe*
- Pandas dataframe with existing table concatenated with new table

**split_test_train_cluster_indices(cluster_group_values, number_of_clusters, clusters_with_no_data, clusters_with_less_than_4, split_percentage)**
- Split data into training and test datasets based on split percentage

Parameters:
cluster_group_values: *1D array of ints*
- List of cluster group assignments generated from test_k_means_cluster_cumulative_returns
number_of_clusters: *int*
- Number of clusters corresponding to k (int)
clusters_with_no_data: *1D array of ints*
- List of the cluster numbers with no data. Can print unique_clusters in the function to see which clusters have values
clusters_with_less_than_4: *1D array of ints*
- List of cluster numbers with less than 4 data points, which is the minimum needed for a 75-25 split
split_percentage: *Float (0-1.0)* 
- Percentage of data assigned to test data, with the rest assigned to training data

Returns:
train_indices_aggregate: *1D array of ints*  
- List of training indices aggregated together, ignoring individual clusters based on split percentage  
test_indices_aggregate: *1D array of ints*  
- List of test indices aggregated together, ignoring individual clusters based on split percentage  
Cluster_group_indices_train: *2D array of ints*  
- List of training indices split based on split percentages for each cluster group. For example, if 0.75 is the training split, each cluster will have 75% of its total data in Cluster_group_indices_train indexed by cluster number  
Cluster_group_indices_test: *2D array of ints*  
- List of test indices split based on split percentages for each cluster group

**perform_regression_main(waveform_biomass_data_file)**
- Custom function based on what regression calculations need to be done.

Parameters:
Waveform_biomass_data_file: *str*  
- String directory link to main table containing waveform and biomass data generated by cumulative_waveform_analysis_main

Returns:
- Result of regression analysis based on the waveform and biomass data

**perform_general_linear_regression(waveform_canopy_heights, biomass_values, waveform_canopy_graph_heights, graph_title)**
- Use scikitlearn’s LinearRegression model to create one linear regression relationship across all data and clusters.

Parameters:
waveform_metric: *n-dimensional np.array*
- Independent/predictor variable data from the waveform in the form of a numpy array with the size of (n_samples, n_features). Generally, a maximum of two features work the best. From our analysis, canopy height is the best predictor of above ground biomass.
biomass_values: *1D array*
- 1D array of biomass values with the same n_samples as waveform_metric
graph_title: *string*
- String for the graph title, change based on the different waveform metric used

Returns:
sorted_waveform_canopy_heights: *n-dimensional np.array*
- Sorted waveform_metric (input) values
sorted_regression_predictions: *n_samples 1D Array*
- Sorted regression predictions based on the linear regression model the size of n_samples

perform_clustered_linear_regression(waveform_canopy_heights, biomass_values, number_of_clusters, cluster_group_values, clusters_with_no_data, clusters_with_limited_data, split_percentage)
- Create a linear regression equation for each cluster’s data

Parameters:
waveform_canopy_heights: *n-dimensional np.array*
Same as waveform_metric in perform_general_linear_regression
biomass_values: *1D array*
- 1D array of biomass values with the same n_samples as waveform_metric
number_of_clusters: *int*
- number of clusters k
cluster_group_values: *1D array of ints*
- List of cluster group assignments generated from test_k_means_cluster_cumulative_returns
clusters_with_no_data: *1D array of ints*
- List of the cluster numbers (int) with no data
clusters_with_limited_data: *1D array of ints*
- List of cluster numbers (int) with less than split_percentage data points (generally less than 4 data points). 

Returns:
Prints out linear regression equation for each cluster along with R^2 score
Graph points used to create the regression

**perform_exponential_regression(waveform_canopy_heights_flattened, biomass_values)**
- Create an exponential regression in the form of a * np.exp(b * t) + c to RH and biomass

Parameters:
waveform_canopy_heights_flattened: *1D array* 
- Waveform canopy heights or another metric
biomass_values: *1D array* 
- Corresponding biomass values

Returns:
Sorted x_input and corresponding exponential predictions

**perform_polynomial_regression(waveform_biomass_data_file)**
Fit a degree 4 polynomial regression function to data

Parameters:
Waveform_biomass_data_file: *str*  
- String directory link to the main table containing waveform and biomass data 

Returns:
- Prints RMSE of the polynomial regression function

**cumulative_waveform_analysis_main(waveform_file, waveform_start_index, waveform_finish_index, biomass_group_num, mapping_tagging_table, apparent_individual_table, waveform_output_file_name)**
- Main waveform processing function to save waveform and biomass into a csv file compatible for regression analysis and clustering functions. Returns a csv file with columns of NEON file number, Pulsenumber, in-situ biomass number associated with waveform, easting, northing, biomass group number, dummy cluster number, and relative elevation at cumulative intensity intervals of 0.025 (waveform data). If saving data to a file, uncomment “save main biomass waveform data” line in the function

waveform_file: *str*
- Directory link to full waveform NEON file
waveform_start_index: *int*  
- Pulsewaves start index to start the analysis on. If any in-situ biomass is found associated with the waveform, the function will save it to the waveform_output_file_name
waveform_finish_index: *int*  
- Pulsewaves end index to conclude analysis
biomass_group_num: *int*  
- Biomass group used for internal tracking 
mapping_tagging_table: *str*  
- Directory link to NEON mapping and tagging table
apparent_individual_table: *str*  
- Directory link to apparent_individual_table table
waveform_output_file_name: *str*  
- File name to save waveforms to. Ensure this group is different for each batch of waveform indices analyzed, so that data isn’t overwritten when saved.

Returns:
- A csv file with the following columns: NEON file number, Pulsenumber, in-situ biomass number associated with waveform, easting, northing, biomass group number, dummy cluster number, and relative elevations at cumulative intensity intervals of 0.025 (waveform data) at each column. The file is saved to the waveform_output_file_name link

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


<!-- CONTACT -->

## Contact

Felix Yu - yu.felix39@gmail.com

Project Link: [https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor](https://github.com/feyu39/Neon-Full-Waveform-LiDAR-Processor)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- Thank you Dr. Jessica Fayne and Dr. Keith Krause for your guidance, support, and contributions to this project. Thank you to the University of Michigan and Dr. Fayne's Remote Sensing Lab for providing the resources and support to make this project possible.

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
