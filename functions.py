'''
Functions
    utility functions mostly for image processing
'''
import numpy as np
from skimage import filters, feature, transform, measure, segmentation, io
import scipy

class DemoImageStream:
    def __init__(self, filepath):
        '''
        Creates a demo stream of a tif stack with shape (t,y,x)
        '''
        self.stack = io.imread(filepath)[:,0]
        self.frame = 0
        self.im_height, self.im_width = self.stack.shape[1], self.stack.shape[2]
    
    def get(self):
        image = self.stack[self.frame]
        self.frame += 1
        if self.frame>=self.stack.shape[0]: self.frame = 0
        return image 

def gaussian_2d(width, height, mu, sigma = 1.0):
    x, y = np.meshgrid(np.linspace(0,1,width)*width, np.linspace(0,1,height)*height)
    g = np.exp(-( ((x-mu[0])**2 + (y-mu[1])**2) / ( 2.0 * sigma**2 ) ) )
    g *= 255/np.max(g)
    g = g.astype("uint8")
    return g

def circle(width, height, center,radius):
    pixels = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if ((j-center[0])**2 + (i-center[1])**2) < radius**2:
                pixels[i,j] = 1
    return pixels

def im_overlay(im1,im2):
    im1,im2 = im1.convert("RGB"),im2.convert("RGB")
    split1,split2 = im1.split(), im2.split()
    merge = 1#Image.merge("RGB", (split1[0],split1[1],Image.blend(split1[2],split2[2],0.05)))
    return merge

def image_to_uint8(image, max_i=None):
    if max_i is not None: assert max_i > -1 and max_i < 256
    else: max_i = 255
    image = image.astype("float64")
    image -= np.min(image)
    if np.max(image)>0:
        image *= max_i/np.max(image)
    image = image.astype("uint8")
    return image

def centroid(array):
    m = measure.moments(array)
    return int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])

def segment_cells(image):
    image = filters.gaussian(image, sigma=4)
    thresholds = filters.threshold_multiotsu(image, classes=3)
    cells = image > thresholds[0]
    distance = scipy.ndimage.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=int(0.4*image.shape[0]), exclude_border=False)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
    return segmented_cells

def segment_cells2(image):
    image = filters.gaussian(image, sigma=4)
    thresholds = filters.threshold_multiotsu(image, classes=4)
    cells = image > thresholds[2]
    distance = scipy.ndimage.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=int(0.3*image.shape[0]), exclude_border=False)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
    return segmented_cells

def segment_cells3(image):
    filtered_image = filters.gaussian(image, sigma=10)
    nuclei = segment_cells2(image)
    x,y = np.ogrid[:image.shape[0],:image.shape[1]]
    size = int(0.3*image.shape[0])
    stitched_image = np.zeros(filtered_image.shape)
    centroids = []
    for i in range(np.max(nuclei)+1):
        if i==0: continue
        #size = 3*np.sqrt(np.sum(nuclei==i))
        point = centroid(nuclei*(nuclei==i))   
        centroids.append(point) 
        qmask = ((x-point[0])**2 < size**2) * ((y-point[1])**2 < size**2)
        #temp_stitch = filtered_image*qmask
        #temp_stitch[temp_stitch==0] = np.min(temp_stitch[temp_stitch.nonzero()])
        #temp_stitch = segment_cells(temp_stitch)
        #stitched_image += (temp_stitch > 0) * (stitched_image == 0)
        stitched_image += (filtered_image * qmask) * (stitched_image == 0)
    stitched_image[stitched_image==0] = np.min(stitched_image[stitched_image.nonzero()])
    segmented_image = segment_cells(stitched_image)
    return segmented_image
    #return stitched_image

def cell_edge(segmented_cells,size):
    border = abs(filters.difference_of_gaussians(segmented_cells,1))
    border = filters.gaussian(border,size)
    border = border > np.mean(border)
    return border

def cell_edge_pattern(segmented_cells,event,size):
    border = abs(filters.difference_of_gaussians(segmented_cells,1))
    border = filters.gaussian(border,size)
    border = border > np.mean(border)
    ill_spot = circle(border.shape[1],border.shape[0],[event.x,event.y],200)
    border = border * ill_spot>0
    return border

def sector_mask(shape, centroid, projection, width):
    """
    Return a boolean mask for a circular sector from centroid of cell in direction of projection with a certain width
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centroid
    direction = projection / np.sqrt(np.sum(projection[:]**2))
    angle = np.arctan2(direction[0], direction[1])
    tmin,tmax = angle - width/2, angle + width/2
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    # circular mask
    circmask = r2 <= shape[0]*shape[1]
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask

def on_egde(image):
    edge_sum = np.sum(image[0,:]) + np.sum(image[:,0]) + np.sum(image[-1,:]) + np.sum(image[:,-1])
    if edge_sum == 0: return False
    else: return True