import urllib
import PIL.Image as Image
import urllib.request as urllib
from ast import literal_eval

import os
import math

class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    # AIzaSyDl3mMCFcSapH5j0TeAM97Ua_noRLiZKJY # project key
    # AIzaSyA7kqDO_6lbirEKKqDHcCIl05-AR0t9Yqg # hw key
    def __init__(self, lat, lng, size=600, zoom=20, type_="satellite", google_key="AIzaSyDl3mMCFcSapH5j0TeAM97Ua_noRLiZKJY"):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = float(lat)
        self._lng = float(lng)
        self._size = size
        self._zoom = zoom
        self._type = type_
        # self._google_key = "AIzaSyBEGgFPYFUUskTcmlFjBXQC3l5SU1xBbw8" #Tinder is overrated
        self._google_key = google_key

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = self._size

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size/ 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1+sin_y)/(1-sin_y)) * -(tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 1
                tile_height:    The number of tiles high the image should be -
                                defaults to 1
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 1)
        tile_height = kwargs.get('tile_height', 1)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None :
            start_x, start_y = self.getXY()

        # Determine the size of the image
        size = self._size
        width, height = size * tile_width, size * tile_height

        #Create a new image of the size require
        map_img = Image.new('RGB', (width,height))

        for x in range(0, tile_width):
            for y in range(0, tile_height) :

                url = "https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size={size1}x{size2}&maptype={type}&key={google_key}".format(lat=self._lat,
                    lng=self._lng, zoom=self._zoom, size1=self._size, size2=self._size, type=self._type, google_key=self._google_key)
                current_tile = str(x)+'-'+str(y)
                urllib.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x*size, y*size))

                os.remove(current_tile)

        return map_img

def getSatelliteImageGoogleMap(data,location_name,row_name):
    data[location_name] = data[location_name].astype(str)
    def inner(lat_long):
        save_name = str(data[data[location_name] == lat_long].iloc[0][row_name])
        download_directory = "data/DOWNLOAD_SATELLITE_HOUSES"
        if not os.path.isdir(download_directory):
            os.mkdir(download_directory)
        dir_path = download_directory + '/' + save_name
        if os.path.isdir(dir_path):
            return
        lat_long = literal_eval(lat_long)
        lat = str(lat_long[0])
        lng = str(lat_long[1])

        # Create a new instance of GoogleMap Downloader
        gmd = GoogleMapDownloader(lat=lat, lng=lng)
        # print("The tile coorindates are {}".format(gmd.getXY()))

        try:
            # Get the high resolution image
            img = gmd.generateImage()
            img.save(dir_path + ".jpg")
        except Exception as e:
            print(e)
            # print(dir_path)
            # print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
            pass
    return inner
