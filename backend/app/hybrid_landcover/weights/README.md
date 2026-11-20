# Panan Lake Sentinel-2 random forest

`panan_s2_2025.xml.gz` is a lossless gzip copy of an OpenCV RTrees classification model. It contains 100 trees and expects a float32 vector of physical surface reflectances in the order `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`. Output IDs 0–5 correspond to `water, bare, grass, wood, crop, built`.

Training used the manually annotated 2025 Panan Lake parent image. The eligible study region contained 121,738 singly labelled pixels. Per-class sampling capped at 6,000 selected 31,788 pixels for the final fit. The upper depth limit was 16. The full-fit model is for new areas. The six development tiles used spatially held-out forests when measuring the hybrid method. Performance across other sites and years has not been measured with this model.

| Artifact | SHA-256 |
| --- | --- |
| Gzip, 5,580,283 bytes | `9825ceb2058f19f341723cba34325e451e84aab68f92b8297bdead5497463f2f` |
| Decompressed original OpenCV XML, 84,803,292 bytes | `186ec01304b7082a11e15b3b0e40aada3d0340e9782d612e8445aa5988097c81` |

The raster reader uses GeoTIFF band descriptions, the dataset `reflectance_scale_factor` tag or GDAL per-band calibration. Untagged floating-point inputs must already contain physical reflectance. NoData, nonfinite samples, and pixels excluded by an ROI or an ignore mask remain outside spectral completion.
