# -*- coding: utf-8 -*-

"""
Created on Mon Aug  1 09:05:30 2023

@author: Robert Masolele
"""

import math
import os
from datetime import datetime

import cv2
import ee
import geemap
import numpy as np
from keras.models import load_model
from skimage import io
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

import rasterio
from rasterio import windows
from retry import retry
from smooth_tiled_predictions import predict_img_with_smooth_windowing

# ee.Authenticate()
ee.Initialize()


# HELPER FUNCTIONS
# Helper functions
def preproc_s1(s1_collection):
    """
    Preprocesses an S1 image collection with slope correction and edge masking

    Parameters
    ----------
    s1_collection : ee.ImageCollection
        An S1 image collection on float/amplitude format (not dB)

    Returns
    -------
    s1_collection : ee.ImageCollection
        The slope-corrected and edge-masked S1 image collection, coverted to dB scaling

    """
    # Do the slope correction
    s1_collection = slope_correction(s1_collection)

    # Mask the edge noise
    s1_collection = s1_collection.map(maskAngGT30)
    s1_collection = s1_collection.map(maskAngLT452)

    # Convert to dB
    s1_collection = s1_collection.map(lin_to_db)

    return ee.ImageCollection(s1_collection)


"""
Code below is adopted from adugnag/gee_s1_ard
"""


def slope_correction(
    collection,
    TERRAIN_FLATTENING_MODEL="VOLUME",
    DEM=ee.Image("USGS/SRTMGL1_003"),
    TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER=0,
):
    """
    Parameters
    ----------
    collection : ee image collection
        DESCRIPTION.
    TERRAIN_FLATTENING_MODEL : string
        The radiometric terrain normalization model, either volume or direct
    DEM : ee asset
        The DEM to be used
    TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER : integer
        The additional buffer to account for the passive layover and shadow
    Returns
    -------
    ee image collection
        An image collection where radiometric terrain normalization is
        implemented on each image
    """

    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        """
        Parameters
        ----------
        theta_iRad : ee.Image
            The scene incidence angle
        alpha_rRad : ee.Image
            Slope steepness in range
        Returns
        -------
        ee.Image
            Applies the volume model in the radiometric terrain normalization
        """

        # Volume model
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        """
        Parameters
        ----------
        theta_iRad : ee.Image
            The scene incidence angle
        alpha_rRad : ee.Image
            Slope steepness in range
        Returns
        -------
        ee.Image
            Applies the direct model in the radiometric terrain normalization
        """
        # Surface model
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = alpha_azRad.cos().multiply(
            (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos()
        )
        return nominator.divide(denominator)

    def _erode(image, distance):
        """

        Parameters
        ----------
        image : ee.Image
            Image to apply the erode function to
        distance : integer
            The distance to apply the buffer
        Returns
        -------
        ee.Image
            An image that is masked to conpensate for passive layover
            and shadow depending on the given distance
        """
        # buffer function (thanks Noel)

        d = (
            image.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )

        return image.updateMask(d.gt(distance))

    def _masking(alpha_rRad, theta_iRad, buffer):
        """
        Parameters
        ----------
        alpha_rRad : ee.Image
            Slope steepness in range
        theta_iRad : ee.Image
            The scene incidence angle
        buffer : TYPE
            DESCRIPTION.
        Returns
        -------
        ee.Image
            An image that is masked to conpensate for passive layover
            and shadow depending on the given distance
        """
        # calculate masks
        # layover, where slope > radar viewing angle
        layover = alpha_rRad.lt(theta_iRad).rename("layover")
        # shadow
        shadow = alpha_rRad.gt(
            ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))
        ).rename("shadow")
        # combine layover and shadow
        mask = layover.And(shadow)
        # add buffer to final mask
        if buffer > 0:
            mask = _erode(mask, buffer)
        return mask.rename("no_data_mask")

    def _correct(image):
        """

        Parameters
        ----------
        image : ee.Image
            Image to apply the radiometric terrain normalization to
        Returns
        -------
        ee.Image
            Radiometrically terrain corrected image
        """

        bandNames = image.bandNames()

        geom = image.geometry()
        proj = image.select(1).projection()

        elevation = DEM.resample("bilinear").reproject(proj, None, 10).clip(geom)

        # calculate the look direction
        heading = ee.Terrain.aspect(image.select("angle")).reduceRegion(
            ee.Reducer.mean(), image.geometry(), 1000
        )

        # in case of null values for heading replace with 0
        heading = ee.Dictionary(heading).combine({"aspect": 0}, False).get("aspect")

        heading = ee.Algorithms.If(
            ee.Number(heading).gt(180),
            ee.Number(heading).subtract(360),
            ee.Number(heading),
        )

        # the numbering follows the article chapters
        # 2.1.1 Radar geometry
        theta_iRad = image.select("angle").multiply(math.pi / 180)
        phi_iRad = ee.Image.constant(heading).multiply(math.pi / 180)

        # 2.1.2 Terrain geometry
        alpha_sRad = ee.Terrain.slope(elevation).select("slope").multiply(math.pi / 180)

        aspect = ee.Terrain.aspect(elevation).select("aspect").clip(geom)

        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)

        phi_sRad = (
            aspect.updateMask(aspect.lte(180))
            .unmask()
            .add(aspect_minus.unmask())
            .multiply(-1)
            .multiply(math.pi / 180)
        )

        # elevation = DEM.reproject(proj,None, 10).clip(geom)

        # 2.1.3 Model geometry
        # reduce to 3 angle
        phi_rRad = phi_iRad.subtract(phi_sRad)

        # slope steepness in range (eq. 2)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        # slope steepness in azimuth (eq 3)
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        # 2.2
        # Gamma_nought
        gamma0 = image.divide(theta_iRad.cos())

        if TERRAIN_FLATTENING_MODEL == "VOLUME":
            # Volumetric Model
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if TERRAIN_FLATTENING_MODEL == "DIRECT":
            scf = _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        # apply model for Gamm0
        gamma0_flat = gamma0.multiply(scf)

        # get Layover/Shadow mask
        mask = _masking(
            alpha_rRad, theta_iRad, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER
        )
        output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)
        output = ee.Image(output).addBands(image.select("angle"), None, True)

        return output.set("system:time_start", image.get("system:time_start"))

    return collection.map(_correct)


def maskAngLT452(image):
    """
    mask out angles >= 45.23993
    Parameters
    ----------
    image : ee.Image
        image to apply the border noise masking
    Returns
    -------
    ee.Image
        Masked image
    """
    ang = image.select(["angle"])
    return image.updateMask(ang.lt(45.23993)).set(
        "system:time_start", image.get("system:time_start")
    )


def maskAngGT30(image):
    """
    mask out angles <= 30.63993
    Parameters
    ----------
    image : ee.Image
        image to apply the border noise masking
    Returns
    -------
    ee.Image
        Masked image
    """

    ang = image.select(["angle"])
    return image.updateMask(ang.gt(30.63993)).set(
        "system:time_start", image.get("system:time_start")
    )


def lin_to_db(image):
    """
    Convert backscatter from linear to dB.
    Parameters
    ----------
    image : ee.Image
        Image to convert
    Returns
    -------
    ee.Image
        output image
    """
    bandNames = image.bandNames().remove("angle")
    db = (
        ee.Image.constant(10)
        .multiply(image.select(bandNames).log10())
        .rename(bandNames)
    )
    return image.addBands(db, None, True)


# def create_s1_composite(bbox, start_date, end_date, pixel_selection, OrbitalPP):
def create_s1_composite(bbox, start_date, end_date, pixel_selection):
    # Create an S1 image mosaic to be added in the bbox (after change)
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filterDate(start_date, end_date)
        .filterBounds(bbox)
    )
    # .filter(ee.Filter.eq('orbitProperties_pass', OrbitalPP))

    # Preprocess the image collection
    s1_preproc = preproc_s1(s1)

    # Select the relevant bands
    s1_preproc = s1_preproc.select("VV", "VH")

    # Create a median composite, a mean composite, or a single image mosaic and multiply with 1000
    if pixel_selection == "median":
        s1_composite = s1_preproc.median()
    elif pixel_selection == "mean":
        s1_composite = s1_preproc.mean()
    elif pixel_selection == "image":
        s1_composite = s1_preproc.sort("system:time_start", False).mosaic()

    return s1_composite


START_DATE = "2024-01-01"
START_DATE2 = "2024-01-01"
END_DATE = "2024-12-31"
END_DATE2 = "2024-12-31"
DESCENDING = "DESCENDING"
ASCENDING = "ASCENDING"

CLOUD_FILTER = 15
CLD_PRB_THRESH = 30
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
bandsused = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


# remove sentinel 1 bands in images with partial coverage
def calculateMaskedPercent(image, band, geom):
    maskedPixelCount = (
        image.select(band)
        .reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=30, maxPixels=1e13
        )
        .get(band)
    )

    totalPixelCount = (
        image.unmask()
        .select(band)
        .reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=30, maxPixels=1e13
        )
        .get(band)
    )

    cloud_cover_roi = (
        ee.Number(1)
        .subtract(ee.Number(maskedPixelCount).divide(totalPixelCount))
        .multiply(100)
    )
    return cloud_cover_roi


# EXTRACT DATA FROM GEE

Tiles2 = ee.FeatureCollection("projects/land-use-292522/assets/Ghana_tiles")

print(len(Tiles2.getInfo()["features"]))


# LOAD THE MODEL
model1 = load_model(
    "Lag_time/model/best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica5.hdf5",
    compile=False,
)


# MAKE PREDICTION

# preprocessing data to  be classified


def normalise_vv(raster):
    raster[raster < -25] = -25
    raster[raster > 0] = 0
    return (raster + 25) / 25


def normalise_vh(raster):
    raster[raster < -30] = -30
    raster[raster > -5] = -5
    return (raster + 30) / 25


def normalise_longitude(raster):
    raster[raster < -180] = -180
    raster[raster > 180] = 180
    return (raster + 180) / 360


def normalise_latitude(raster):
    raster[raster < -60] = -60
    raster[raster > 60] = 60
    return (raster + 60) / 120


def normalise_altitude(raster):
    raster[raster < -400] = -400
    raster[raster > 8000] = 8000
    return (raster + 400) / 8400


def normalise_ndre(raster):
    raster[raster < -1] = -1
    raster[raster > 1] = 1
    return (raster + 1) / 2


def normalise_evi(raster):
    raster[raster < -1] = -1
    raster[raster > 1] = 1
    return (raster + 1) / 2


def norm(image):
    NORM_PERCENTILES = np.array(
        [
            [1.7417268007636313, 2.023298706048351],
            [1.7261204997060209, 2.038905204308012],
            [1.6798346251414997, 2.179592821212937],
            [2.3828939530384052, 2.7578332604178284],
            [1.7417268007636313, 2.023298706048351],
            [1.7417268007636313, 2.023298706048351],
            [1.7417268007636313, 2.023298706048351],
            [1.7417268007636313, 2.023298706048351],
            [1.7417268007636313, 2.023298706048351],
        ]
    )

    image = np.log(image * 0.005 + 1)
    image = (image - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]

    # Get a sigmoid transfer of the re-scaled reflectance values.
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)

    return image


def extract_lat_lon(image_path):
    """
    Extract latitude and longitude values for each pixel in a georeferenced image.

    Args:
        image_path (str): Path to the Sentinel image file.

    Returns:
        latitudes (numpy array): Array of latitude values for each pixel.
        longitudes (numpy array): Array of longitude values for each pixel.
    """
    # Open the Sentinel image using rasterio
    with rasterio.open(image_path) as src:
        # Get the affine transform and CRS information
        transform = src.transform
        crs = src.crs

        # Get the dimensions of the image
        height, width = src.shape

        # Create arrays for row and column indices
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Compute the geographic coordinates
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs = np.array(xs)
        ys = np.array(ys)

        # Transform to latitude and longitude if needed
        if crs.to_string() != "EPSG:4326":  # EPSG:4326 is WGS84 (lat/lon)
            from pyproj import Transformer

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            longitudes, latitudes = transformer.transform(xs, ys)
        else:
            longitudes, latitudes = xs, ys

    return latitudes, longitudes


start2 = datetime.now()


def preprocess_planet(x_img, image_path):
    latitudes, longitudes = extract_lat_lon(image_path)

    # x_img = x_img[:,:,imgbands]
    x_img1 = x_img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    # x_img1 =np.where(x_img1<0, 0, x_img1)

    x_img2 = norm(x_img1)
    vv = normalise_vv(x_img[:, :, 9])
    vh = normalise_vh(x_img[:, :, 10])
    # ratio = (vh/vv)#[:,:,np.newaxis]
    alt = normalise_altitude(x_img[:, :, 11])
    lon = normalise_longitude(longitudes)
    lat = normalise_latitude(latitudes)  # 7

    SIZE_X = x_img.shape[0]
    SIZE_Y = x_img.shape[1]
    red_edge1 = x_img1[:, :, 3]
    nir = x_img1[:, :, 6]
    red = x_img1[:, :, 2]
    green = x_img1[:, :, 1]
    blue = x_img1[:, :, 0]

    ndvi = np.where((nir + red) == 0.0, 0, (nir - red) / (nir + red))
    evi = np.where(
        (nir + red) == 0.0, 0, 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    )
    evi = normalise_evi(evi)
    ndre = np.where((nir + red_edge1) == 0.0, 0, (nir - red_edge1) / (nir + red_edge1))
    ndre = normalise_ndre(ndre)

    ndvi = np.reshape(ndvi, (SIZE_X, SIZE_Y, 1))
    evi = np.reshape(evi, (SIZE_X, SIZE_Y, 1))
    ndre = np.reshape(ndre, (SIZE_X, SIZE_Y, 1))
    vv = np.reshape(vv, (SIZE_X, SIZE_Y, 1))
    vh = np.reshape(vh, (SIZE_X, SIZE_Y, 1))
    alt = np.reshape(alt, (SIZE_X, SIZE_Y, 1))
    lon = np.reshape(lon, (SIZE_X, SIZE_Y, 1))
    lat = np.reshape(lat, (SIZE_X, SIZE_Y, 1))
    image = np.concatenate((x_img2, ndvi, ndre, evi, vv, vh, alt, lon, lat), axis=2)
    image = np.nan_to_num(image)
    return image


def preprocess_planet2(x_img):
    x_img = x_img[:, :, imgbands2]
    x_img1 = x_img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    x_img1 = np.where(x_img1 < 0, 0, x_img1)

    x_img1 = norm(x_img1)
    alt = normalise_altitude(x_img[:, :, 9])
    lon = normalise_longitude(x_img[:, :, 10])
    lat = normalise_latitude(x_img[:, :, 11])

    SIZE_X = x_img.shape[0]
    SIZE_Y = x_img.shape[1]
    nir = x_img[:, :, 6]
    red = x_img[:, :, 2]
    green = x_img[:, :, 1]
    blue = x_img[:, :, 0]
    ndvi = np.where((nir + red) == 0.0, 0, (nir - red) / (nir + red))
    ndvi = np.reshape(ndvi, (SIZE_X, SIZE_Y, 1))
    alt = np.reshape(alt, (SIZE_X, SIZE_Y, 1))
    lon = np.reshape(lon, (SIZE_X, SIZE_Y, 1))
    lat = np.reshape(lat, (SIZE_X, SIZE_Y, 1))
    image = np.concatenate((x_img1, ndvi, alt, lon, lat), axis=2)
    return image


# pad images
def pad_image(image, patch_size, padding_mode="reflect"):
    """
    Pads the image using the specified mode.
    Args:
        image: Input image (H, W, C)
        patch_size: Tuple (patch_height, patch_width)
        padding_mode: 'reflect', 'symmetric', or 'constant'
    Returns:
        padded_image, pad_values (used for cropping back)
    """
    pad_h = patch_size[0] // 2
    pad_w = patch_size[1] // 2

    padded_image = np.pad(
        image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=padding_mode
    )

    return padded_image, (pad_h, pad_w)


# Crop Back to Original Image Size
def crop_back(predicted_image, pad_values):
    """
    Crops the predicted image back to the original size.
    """
    pad_h, pad_w = pad_values
    return predicted_image[pad_h:-pad_h, pad_w:-pad_w, :]


# function to extract each tiles coordinates
featlist = Tiles2.getInfo()["features"]


# print(len(featlist))
def unpack(thelist):
    unpacked = []
    for i in thelist:
        unpacked.append(i[0])
        unpacked.append(i[1])
    return unpacked


# extract the image of each tile by using GEEMAP package
# def tile(root_directory,image, featlist, year=0, filename1=0):
@retry(tries=50, delay=1, backoff=2)
def tile(root_directory, featlist, year=0, filename1=0, buffer_size=0, off1=0, off2=0):
    filee = root_directory + "planet2_" + str(year)
    if not os.path.exists(filee):
        #  print(filee + " already created")
        # else:
        os.mkdir(filee)
    for f in featlist:
        startp = datetime.now()
        geomlist = None
        geomlist = unpack(f["geometry"]["coordinates"][0])
        year = str(year)
        feat = ee.Geometry.Polygon(geomlist)
        disS = f["properties"]["Tile_ID"]

        """IGNORE TILES WITHOUT LOSS PIXELS IN GEE"""
        # Deforest1   = FinalIMG2023.select(['loss'])
        # maxReducer = ee.Reducer.max()
        # theMax = Deforest1.reduceRegion(maxReducer, feat.bounds())
        # flag = np.array((ee.Array(theMax.get("loss")).getInfo()))
        # print(flag)
        # # extract a number array from this region
        # #flag = geemap.ee_to_numpy(Deforest1)
        # #flag = ee.Algorithms.IsEqual(theMax, 0)
        # if flag==0:
        #     print('No deforested pixel in tile')
        # else:

        sent2023 = create_s1_composite(
            feat.bounds().buffer(buffer_size).bounds(), START_DATE, END_DATE, "median"
        )
        sent2023_2 = create_s1_composite(
            feat.bounds().buffer(buffer_size).bounds(), START_DATE2, END_DATE2, "median"
        )

        Longitude = (
            ee.Image.pixelLonLat()
            .select("longitude")
            .clip(feat.bounds().buffer(buffer_size).bounds())
        )
        Latitude = (
            ee.Image.pixelLonLat()
            .select("latitude")
            .clip(feat.bounds().buffer(buffer_size).bounds())
        )  # .buffer(250).bounds()
        Elevation = (
            ee.ImageCollection("COPERNICUS/DEM/GLO30")
            .select("DEM")
            .mosaic()
            .clip(feat.bounds().buffer(buffer_size).bounds())
        )

        # import sentinel 2 data
        # Define variables to form ImageCollection
        # start/end date, bands to use, and parameters for cloud masking

        SA = feat.bounds().buffer(buffer_size).bounds()

        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        csPlus = ee.ImageCollection(
            "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
        ).filterDate("2024-01-01", "2024-12-30")

        # QA_BAND = 'cs';
        QA_BAND = "cs_cdf"
        CLEAR_THRESHOLD = 0.40

        BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

        def clearMask(img):
            img = (
                img.toFloat()
                .resample("bilinear")
                .reproject(img.select("B2").projection())
            )
            return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

        basemap23 = (
            s2.filterBounds(SA)
            .filterDate("2024-01-01", "2024-12-30")
            .linkCollection(csPlus, [QA_BAND])
            .map(clearMask)
            .median()
            .clip(SA)
            .select(BANDS)
        )

        # Add forest loss from hansen forest loss data
        loss22 = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(
            feat.bounds().buffer(buffer_size).bounds()
        )
        treeLoss = loss22.select(["loss"])

        sent = sent2023_2

        image = (
            basemap23.addBands(sent)
            .addBands(Elevation)
            .addBands(Longitude)
            .addBands(Latitude)
            .addBands(treeLoss)
            .int16()
        )

        filename1 = root_directory + "planet2_" + year + "/" + str(disS) + ".tif"
        filename2 = root_directory + "PredictedD_AF2" + "/" + str(disS) + "_georef.tif"
        filename3 = root_directory + "conf" + "/" + str(disS) + "_georef.tif"

        if os.path.exists(filename2):
            print(filename2 + " already downloaded")
        else:
            print("downloading " + str(disS))
            startd = datetime.now()
            import logging

            logger = logging.getLogger()

            try:
                geemap.ee_export_image(
                    image,
                    filename=filename1,
                    scale=10,
                    region=feat.bounds().buffer(buffer_size).bounds(),
                    # file_per_band=False
                )
            except Exception as e:
                print(e)

            stopd = datetime.now()
            # Execution time
            execution_time_download = stopd - startd
            print("Download execution time is: ", execution_time_download)
            print("Next step...")
            img_dir = root_directory + "planet2_" + year + "/"
            Pred = root_directory + "PredictedD_AF2" + "/"
            conf_dir = root_directory + "conf" + "/"
            PredFull_dir = root_directory + "PredFull" + "/"
            prob_dir = root_directory + "prob" + "/"
            probCocoa_dir = root_directory + "probCocoa" + "/"
            probCoffee_dir = root_directory + "probCoffee" + "/"

            # prob_dir = root_directory+"prob"+"/"

            if not os.path.exists(img_dir):
                # print(img_dir+ " already created")
                # else:
                os.mkdir(img_dir)
            if not os.path.exists(Pred):
                # print(Pred+ " already created")
                # else:
                os.mkdir(Pred)

            if not os.path.exists(conf_dir):
                # print(Pred+ " already created")
                # else:
                os.mkdir(conf_dir)

            if not os.path.exists(PredFull_dir):
                os.mkdir(PredFull_dir)

            if not os.path.exists(prob_dir):
                os.mkdir(prob_dir)

            if not os.path.exists(probCocoa_dir):
                os.mkdir(probCocoa_dir)

            if not os.path.exists(probCoffee_dir):
                os.mkdir(probCoffee_dir)

            # size of patches
            patch_size = 64
            # Number of classes
            n_classes = 25
            start2 = datetime.now()

            try:
                x_img = io.imread(
                    root_directory + "planet2_" + year + "/" + str(disS) + ".tif"
                )  # Read each image

            except FileNotFoundError as F:
                print(F)
                pass
            except UnboundLocalError as L:
                print(L)
                pass
            else:
                if x_img.shape[2] == 15:
                    try:
                        print(x_img.shape)
                        loss = x_img[:, :, 14].astype(
                            "uint8"
                        )  # index 8 is out of bounds for axis 2 with size 7
                        print("image pre-processing and prediction 1")

                    except UnboundLocalError as L:
                        print(L)

                    else:
                        x_img3 = preprocess_planet(x_img, filename1)
                        ########## preprocessing
                        ########
                        # Extract image for prediction band 1 to 4
                        print(x_img3.shape)

                        ############################################################################32 patch size ###############################################################################

                        # predict using smooth blending

                        patch_size2 = (64, 64)  # Model input size
                        padded_image, pad_values = pad_image(x_img3, patch_size2)
                        print("Padded Image Shape:", padded_image.shape)

                        startp = datetime.now()
                        predictions_smooth2 = predict_img_with_smooth_windowing(
                            padded_image,
                            window_size=patch_size,
                            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                            nb_classes=n_classes,
                            pred_func=(
                                # lambda img_batch_subdiv: model1.predict((img_batch_subdiv))
                                lambda img_batch_subdiv: model1.predict_on_batch(
                                    img_batch_subdiv
                                )
                            ),
                        )
                        stopp = datetime.now()
                        # Execution time
                        execution_time_prediction = stopp - startp
                        print(
                            "prediction execution time is: ", execution_time_prediction
                        )

                        predictions_smooth = crop_back(predictions_smooth2, pad_values)
                        print("Final Prediction Shape:", predictions_smooth.shape)

                        final_predictionF = np.argmax(predictions_smooth, axis=2)
                        # print(final_prediction.shape)

                        # final_prediction = np.where(final_prediction==0, 6, final_prediction)

                        final_prediction = np.where(loss == 0, 0, final_predictionF)
                        ##predictions_smooth = np.where(mask==0, 0, predictions_smooth[:,:,0]& predictions_smooth[:,:,1])
                        ##print(final_prediction.shape)

                        conf = np.subtract(
                            np.max(predictions_smooth, axis=2),
                            np.partition(predictions_smooth, -2, axis=2)[:, :, -2],
                        )
                        conf = np.round(conf * 100).astype(np.uint8)
                        # print(conf.shape)

                        # conf = np.where(loss==0, 0, conf)
                        # print(conf.shape)
                        loss2 = np.reshape(loss, (loss.shape[0], loss.shape[1], 1))

                        # Max probabilities
                        prob = np.max(predictions_smooth, axis=2)
                        prob = np.reshape(prob, (prob.shape[0], prob.shape[1], 1))
                        prob = np.round(prob * 100).astype(np.uint8)

                        # prob for cocoa
                        probCocoa = predictions_smooth[:, :, 13]
                        probCocoa = np.reshape(
                            probCocoa, (probCocoa.shape[0], probCocoa.shape[1], 1)
                        )
                        probCocoa = np.round(probCocoa * 100).astype(np.uint8)

                        # prob for cocoa
                        probCoffee = predictions_smooth[:, :, 8]
                        probCoffee = np.reshape(
                            probCoffee, (probCoffee.shape[0], probCoffee.shape[1], 1)
                        )
                        probCoffee = np.round(probCoffee * 100).astype(np.uint8)

                        # georeferencing pred
                        filename1 = (
                            root_directory
                            + "planet2_"
                            + year
                            + "/"
                            + str(disS)
                            + ".tif"
                        )
                        path_to_img_pred = Pred + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_pred, final_prediction)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        a = rasterio.open(path_to_img_pred, dtype="uint8")
                        slice_ = (
                            slice(off1, a.height - off2),
                            slice(off1, a.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)
                        # print(a.profile)
                        predimg = a.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # profile = src.profile.copy()
                            # print(src)
                            # print(src.profile)

                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()

                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # predimg.shape[0],
                                    "width": window_slice.width,  # predimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            Pred + str(disS) + "_georef.tif", "w", **profile
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, predimg)
                        a.close()
                        os.remove(Pred + str(disS) + ".tif")

                        # georeference full predicted image
                        # georeferencing
                        path_to_img_predFull_dir = PredFull_dir + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_predFull_dir, final_predictionF)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        b = rasterio.open(path_to_img_predFull_dir, dtype="uint8")
                        slice_ = (
                            slice(off1, b.height - off2),
                            slice(off1, b.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)

                        # print(a.profile)
                        predFimg = b.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # profile = src.profile.copy()
                            # print(src)
                            # print(src.profile)
                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()
                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # predFimg.shape[0],
                                    "width": window_slice.width,  # predFimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            PredFull_dir + str(disS) + "_georef.tif", "w", **profile
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, predFimg)
                        b.close()
                        os.remove(PredFull_dir + str(disS) + ".tif")

                        # georeferencing conf
                        path_to_img_conf_dir = conf_dir + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_conf_dir, conf)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        c = rasterio.open(path_to_img_conf_dir, dtype="uint8")
                        slice_ = (
                            slice(off1, c.height - off2),
                            slice(off1, c.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)
                        # print(a.profile)
                        confimg = c.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # slice_ = (slice(11,src.height-10),slice(11,src.width-10))
                            # window_slice = windows.Window.from_slices(*slice_)
                            # src = src.read(1, out_dtype='uint8', window=window_slice)
                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()
                            # print(src)
                            # print(src.profile)

                            # slice_ = (slice(11,src.height-10),slice(11,src.width-10))
                            # window_slice = windows.Window.from_slices(*slice_)

                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # confimg.shape[0],
                                    "width": window_slice.width,  # confimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            conf_dir + str(disS) + "conf_georef.tif", "w", **profile
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, confimg)
                        c.close()
                        os.remove(conf_dir + str(disS) + ".tif")

                        # georeferencing prob
                        path_to_img_prob_dir = prob_dir + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_prob_dir, prob)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        d = rasterio.open(path_to_img_prob_dir, dtype="uint8")
                        slice_ = (
                            slice(off1, d.height - off2),
                            slice(off1, d.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)
                        # print(a.profile)
                        probimg = d.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # profile = src.profile.copy()
                            # print(src)
                            # print(src.profile)
                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()
                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # probimg.shape[0],
                                    "width": window_slice.width,  # probimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            prob_dir + str(disS) + "prob_georef.tif", "w", **profile
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, probimg)
                        d.close()
                        os.remove(prob_dir + str(disS) + ".tif")

                        # os.remove(root_directory+'planet2_'+year+'/'+str(disS)+'.tif')

                        # georeferencing probCocoa
                        path_to_img_probCocoa_dir = probCocoa_dir + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_probCocoa_dir, probCocoa)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        e = rasterio.open(path_to_img_probCocoa_dir, dtype="uint8")
                        slice_ = (
                            slice(off1, e.height - off2),
                            slice(off1, e.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)
                        # print(a.profile)
                        probimgCocoa = e.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # profile = src.profile.copy()
                            # print(src)
                            # print(src.profile)
                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()
                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # probimg.shape[0],
                                    "width": window_slice.width,  # probimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            probCocoa_dir + str(disS) + "prob_georef.tif",
                            "w",
                            **profile,
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, probimgCocoa)
                        e.close()
                        os.remove(probCocoa_dir + str(disS) + ".tif")

                        # os.remove(root_directory+'planet2_'+year+'/'+str(disS)+'.tif')

                        # georeferencing probCoffee
                        path_to_img_probCoffee_dir = probCoffee_dir + str(disS) + ".tif"
                        cv2.imwrite(path_to_img_probCoffee_dir, probCoffee)
                        # raster = gdal.Open(path+image_name)
                        # array2raster(image_name2+'PRED_GEOREF.tif', raster, final_prediction, "uint16")
                        e = rasterio.open(path_to_img_probCoffee_dir, dtype="uint8")
                        slice_ = (
                            slice(off1, e.height - off2),
                            slice(off1, e.width - off2),
                        )
                        window_slice = windows.Window.from_slices(*slice_)
                        # print(a.profile)
                        probimgCoffee = e.read(
                            1, out_dtype="uint8", window=window_slice
                        )  # read the entire array
                        with rasterio.open(filename1, count=1, dtype="uint8") as src:
                            # profile = src.profile.copy()
                            # print(src)
                            # print(src.profile)
                            aff = rasterio.windows.transform(
                                window_slice, src.transform
                            )
                            profile = src.meta.copy()
                            # aff = src.transform
                            profile.update(
                                {
                                    "dtype": "uint8",
                                    "height": window_slice.height,  # probimg.shape[0],
                                    "width": window_slice.width,  # probimg.shape[1],
                                    "count": 1,
                                    "nodata": 0,
                                    "transform": aff,
                                }
                            )
                        # print('georeferencing now')
                        with rasterio.open(
                            probCoffee_dir + str(disS) + "prob_georef.tif",
                            "w",
                            **profile,
                        ) as dst:
                            # print('Its all well')
                            dst.write_band(1, probimgCoffee)
                        e.close()
                        os.remove(probCoffee_dir + str(disS) + ".tif")

                        os.remove(
                            root_directory
                            + "planet2_"
                            + year
                            + "/"
                            + str(disS)
                            + ".tif"
                        )


# image download, prediction and georeferencing
root_directory = "Latin_America_PredGhana/"

tile(
    root_directory, featlist=featlist, year=2024, buffer_size=1, off1=0, off2=0
)  # 50, 11, 10

stop2 = datetime.now()
# Execution time
execution_time_download = stop2 - start2
print(
    "Data download, prediction and georeferencing execution time is: ",
    execution_time_download,
)
