import math
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    right_lines_t_x=list()
    right_lines_b_x=list()
    
    left_lines_t_x=list()
    left_lines_b_x=list()
    
    right_lines_t_y=list()
    right_lines_b_y=list()
    
    left_lines_t_y=list()
    left_lines_b_y=list()
 
    
    for line in lines:
        print(line)
        for x1, y1, x2, y2 in line:
            slope=((y2-y1)/(x2-x1))
            
            if slope<0:
                left_lines_t_x.append(x1)
                left_lines_b_x.append(x2)
                left_lines_t_y.append(y1)
                left_lines_b_y.append(y2)
            else:
                right_lines_t_x.append(x1)
                right_lines_b_x.append(x2)
                right_lines_t_y.append(y1)
                right_lines_b_y.append(y2)
    
    l_top=[sum(left_lines_t_x)/len(left_lines_t_x), sum(left_lines_t_y)/len(left_lines_t_y)]
    l_bottom=[sum(left_lines_b_x)/len(left_lines_b_x), sum(left_lines_b_y)/len(left_lines_b_y)]
    
    r_top=[sum(right_lines_t_x)/len(right_lines_t_x), sum(right_lines_t_y)/len(right_lines_t_y)]
    r_bottom=[sum(right_lines_b_x)/len(right_lines_b_x), sum(right_lines_b_y)/len(right_lines_b_y)]
    
    print(l_top)
    print(l_bottom)
    
    #find the two max and min x points
    
    m_l=(l_top[1]-l_bottom[1])/(l_top[0]-l_bottom[0])
    b_l=l_top[1]-m_l*l_top[0]
    
    m_r=(r_top[1]-r_bottom[1])/(r_top[0]-r_bottom[0])
    b_r=r_top[1]-m_r*r_top[0]
    
    
    y_mid=int(img.shape[0]/2)+50
    y_bottom=img.shape[0]
    
    l_t=[int((y_mid-b_l)/m_l), y_mid]
    l_b=[int((y_bottom-b_l)/m_l), y_bottom]
    
    r_t=[int((y_mid-b_r)/m_r), y_mid]
    r_b=[int((y_bottom-b_r)/m_r), y_bottom]
    
    
    
    cv2.line(img, (l_t[0], l_t[1]), (l_b[0], l_b[1]), color, thickness)
    cv2.line(img, (r_t[0], r_t[1]), (r_b[0], r_b[1]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


    # TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

image = mpimg.imread('test_images/solidWhiteRight.jpg')
grey=grayscale(image)

blur_grey=gaussian_blur(grey, 5)

low_threshold = 50
high_threshold = 150

edges=canny(blur_grey, low_threshold, high_threshold)

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(imshape[1]/2-5, imshape[0]/2+50), (imshape[1]/2+5, imshape[0]/2+50), (imshape[1],imshape[0])]], dtype=np.int32)

masked_img=region_of_interest(edges, vertices)

rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20    # minimum number of votes (intersections in Hough grid cell)
min_line_len = 50 #minimum number of pixels making up a line
max_line_gap = 30 # maximum gap in pixels between connectable line segments

line_img=hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)

result_img=weighted_img(line_img, image,)

plt.imshow(result_img)
mpimg.imsave("solidWhiteRight-after.jpg", result_img)