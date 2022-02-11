'''
Model
    Model to be imported in the ModelThread of the main program
    parameters can be accessed through the dictionaries, 
        this structure is needed to make the synchronization with the ModelThread independent of number and names of parameters
'''

from functions import *
import numpy as np
from skimage import measure, filters 

class AutomaticDirected():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.area_check = False

        # controlled parameters
        self.controlled_parameters = {
            "direction": np.array([0,0]),
            "shape": 1.5,
            "id": np.array([0,0]),
            "id_set": False
        }
        self.parameter_controls = {
            "double_left": "direction",
            "double_left_set": None,
            "textbox": "shape",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        self.dmd_image = self.create_directed_pattern(self.controlled_parameters["shape"])
    
    def create_directed_pattern(self, width):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
        projection = self.controlled_parameters["direction"] - self.internal_parameters["centroid"]
        print(self.internal_parameters["segment_id"], self.internal_parameters["centroid"])
        return cell_edge(self.processed_image,25)*sector_mask(self.processed_image.shape,self.internal_parameters["centroid"], projection, width)*(self.processed_image==self.internal_parameters["segment_id"])

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class AutomaticDirectedOnOff():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.area_check = False
        self.on = True
        self.frame_count = 0

        # controlled parameters
        self.controlled_parameters = {
            "direction": np.array([0,0]),
            "shape": 1.5,
            "id": np.array([0,0]),
            "id_set": False
        }
        self.parameter_controls = {
            "double_left": "direction",
            "double_left_set": None,
            "textbox": "shape",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        if self.frame_count == 40 and self.on: 
            self.on = not self.on
            self.frame_count = 0
        if self.frame_count == 20 and not self.on: 
            self.on = not self.on
            self.frame_count = 0
        self.frame_count += 1
        self.dmd_image = self.create_directed_pattern(self.controlled_parameters["shape"])
        #self.dmd_image = circle(self.processed_image.shape[1],self.processed_image.shape[0], [200,200],20)
    
    def create_directed_pattern(self, width):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.frame_count = 0
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
        projection = self.controlled_parameters["direction"] - self.internal_parameters["centroid"]
        return self.on*cell_edge(self.processed_image,25)*sector_mask(self.processed_image.shape,self.internal_parameters["centroid"], projection, width)*(self.processed_image==self.internal_parameters["segment_id"])

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class ManualRepositioning():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.area_check = False

        # controlled parameters
        self.controlled_parameters = {
            "line_start": np.array([0,0]),
            "line_end": np.array([100,100]),
            "line_set": False,
            "id": np.array([0,0]),
            "id_set": False
        }
        self.parameter_controls = {
            "left": "line_start",
            "left_release": "line_end",
            "left_release_set": "line_set",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1,
            "line_mask": np.ones((1030,1300))
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        self.dmd_image = self.create_half_pattern()
    
    def create_half_pattern(self):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])

        if self.controlled_parameters["line_set"]:
            x1, x2 = self.controlled_parameters["line_start"], self.controlled_parameters["line_end"]
            x,y = np.ogrid[:self.processed_image.shape[0],:self.processed_image.shape[1]]
            d = (x-x1[0])*(x2[1]-x1[1]) - (y-x1[1])*(x2[0]-x1[0])
            self.internal_parameters["line_mask"] = d > 0 
            self.controlled_parameters["line_set"] = False
        else:
            movement = self.internal_parameters["centroid"] - centroid
            self.controlled_parameters["line_start"] += movement
            self.controlled_parameters["line_end"] += movement
            x1, x2 = self.controlled_parameters["line_start"], self.controlled_parameters["line_end"]
            x,y = np.ogrid[:self.processed_image.shape[0],:self.processed_image.shape[1]]
            d = (x-x1[0])*(x2[1]-x1[1]) - (y-x1[1])*(x2[0]-x1[0])
            self.internal_parameters["line_mask"] = d > 0 
            self.controlled_parameters["line_set"] = False

        return (self.processed_image==self.internal_parameters["segment_id"])*self.internal_parameters["line_mask"]

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class Test():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.area_check = False
        self.led_power = 10
        self.led_set = True
        self.count = 0

        # controlled parameters
        self.controlled_parameters = {
            "direction": np.array([0,0]),
            "shape": 1.5,
            "id": np.array([0,0]),
            "id_set": False
        }
        self.parameter_controls = {
            "double_left": "direction",
            "double_left_set": None,
            "textbox": "shape",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        #print("image:", np.max(cam_image), "led:", self.led_power)
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        self.count += 1
        if self.led_set: self.led_set = False
        #if self.count % 5 == 0: 
        #    self.led_power += 10
        #    self.led_set = True
        self.dmd_image = circle(self.processed_image.shape[1],self.processed_image.shape[0], [500,500], 50)
    
    def create_directed_pattern(self, width):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
        projection = self.controlled_parameters["direction"] - self.internal_parameters["centroid"]
        return cell_edge(self.processed_image,25)*sector_mask(self.processed_image.shape,self.internal_parameters["centroid"], projection, width)*(self.processed_image==self.internal_parameters["segment_id"])

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class AutomaticDirectedTurn():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction and changes direction after a set number of frames
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.area_check = False
        self.on = True
        self.frame_count = 0
        #self.led_power = 1
        #self.led_set = True

        # controlled parameters
        self.controlled_parameters = {
            "direction": np.array([0,0]),
            "shape": 1.5,
            "id": np.array([0,0]),
            "id_set": False
        }
        self.parameter_controls = {
            "double_left": "direction",
            "double_left_set": None,
            "textbox": "shape",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1,
            "angle": 0.01
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        if self.frame_count == 40 and self.on: 
            #self.on = not self.on
            #if self.internal_parameters["angle"] < 1.0:
            #    self.internal_parameters["angle"] = 2 * self.internal_parameters["angle"]
            #else:
            #    self.internal_parameters["angle"] += 1.28
            self.internal_parameters["angle"] += 0.8
            self.frame_count = 0
        #if self.frame_count == 20 and not self.on: 
        #    self.on = not self.on
        #    self.frame_count = 0
        #if self.frame_count % 2 == 0: self.led_power = 0
        #else: self.led_power = 1
        self.frame_count += 1
        if on_egde(self.dmd_image): self.internal_parameters["angle"] += 3.1415
        self.dmd_image = self.create_directed_pattern(self.controlled_parameters["shape"])
    
    def create_directed_pattern(self, width):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.frame_count = 0
            self.internal_parameters["angle"] = 0.2
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
        projection = self.controlled_parameters["direction"] - self.internal_parameters["centroid"]
        theta = self.internal_parameters["angle"]
        projection = np.dot(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]), projection)
        return self.on*cell_edge(self.processed_image,25)*sector_mask(self.processed_image.shape,self.internal_parameters["centroid"], projection, width)*(self.processed_image==self.internal_parameters["segment_id"])

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class ManualLEXY():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction and changes direction after a set number of frames
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.on = False
        self.frame_count = 0
        self.led_power = 1
        self.led_set = False

        # controlled parameters
        self.controlled_parameters = {
            "id": np.array([0,0]),
            "id_set": False,
            "led_set": False,
            "led_power": 1
        }
        self.parameter_controls = {
            "double_left": "id",
            "double_left_set": "id_set",
            "textbox": "led_power",
            "right": None,
            "right_set": "led_set"
        }

        # internal parameters
        self.internal_parameters = {
            "segment_id": 1,
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.controlled_parameters["id_set"]:
            temp = segment_cells2(cam_image)
            temp_edge = cell_edge(temp, 7)
            temp = temp * (temp_edge == False)
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = temp[id_x,id_y]
            self.processed_image = (temp == self.internal_parameters["segment_id"])
            #self.processed_image = segment_cells2(cam_image)
            print("check:",np.sum(self.processed_image))
            self.on = True
            self.led_set = True
            self.controlled_parameters["id_set"] = False
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        self.led_power = int(self.controlled_parameters["led_power"])
        if self.controlled_parameters["led_set"]:
            self.controlled_parameters["led_set"] = False
            self.led_set = not self.led_set
            self.led_power = 0
            self.on = not self.on 
        self.dmd_image = self.on * self.processed_image
    
    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class SemiManualLEXY():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction and changes direction after a set number of frames
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((680,600))
        self.on = False
        self.frame_count = 0
        self.led_power = 1
        self.led_set = False

        # controlled parameters
        self.controlled_parameters = {
            "id": np.array([0,0]),
            "id_set": False,
            "led_set": False,
            "led_power": 1
        }
        self.parameter_controls = {
            "double_left": "id",
            "double_left_set": "id_set",
            "textbox": "led_power",
            "right": None,
            "right_set": "led_set"
        }

        # internal parameters
        self.internal_parameters = {
            "segment_id": 1,
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.controlled_parameters["id_set"]:
            temp = segment_cells2(cam_image)
            temp_edge = cell_edge(temp, 7)
            temp = temp * (temp_edge == False)
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = temp[id_x,id_y]
            self.processed_image = (temp == self.internal_parameters["segment_id"])
            #self.processed_image = segment_cells2(cam_image)
            print("check:",np.sum(self.processed_image))
            self.on = True
            self.led_set = True
            self.controlled_parameters["id_set"] = False
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        self.led_power = int(self.controlled_parameters["led_power"])
        if self.controlled_parameters["led_set"]:
            self.controlled_parameters["led_set"] = False
            #self.led_set = not self.led_set
            #self.led_power = 0
            self.frame_count = 0
            self.on = False
        if self.frame_count == 5:
            self.led_set = True
            self.on = True
        if self.frame_count == 85:
            self.on = False
            self.led_power = 0
        if not self.on: self.led_power = 0
        self.dmd_image = self.on * self.processed_image
        self.frame_count +=1
    
    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class AutomaticPath():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((684,608))
        self.area_check = False

        # controlled parameters
        self.controlled_parameters = {
            "direction": np.array([0,0]),
            "shape": 1.5,
            "id": np.array([0,0]),
            "id_set": False,
            "path_set": False
        }
        self.parameter_controls = {
            "double_left": None,
            "double_left_set": "path_set",
            "textbox": "shape",
            "right": "id",
            "right_set": "id_set"
        }

        # internal parameters
        self.internal_parameters = {
            "centroid": np.array([0,0]),
            "segment_id": 1,
            "path": self.create_path(),
            "pp_id": None, # path point ID 
            "L2": 0 # point choice distance squared
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.area_check:
            new_segmentation = segment_cells(cam_image)
            area_old = np.sum(self.processed_image==self.internal_parameters["segment_id"])
            centroid = self.internal_parameters["centroid"]
            new_id = new_segmentation[centroid[0],centroid[1]]
            area_new = np.sum(new_segmentation==new_id)
            if area_new > 0.7*area_old and area_new < 1.3*area_old:
                self.processed_image = new_segmentation
        else: 
            self.area_check = not self.area_check
            self.processed_image = segment_cells(cam_image)
    
    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        if self.controlled_parameters["path_set"]:
            n_points = self.internal_parameters["path"].shape[0]
            if self.internal_parameters["pp_id"] is None:
                distance = 999999
                for i in range(n_points):
                    temp = self.internal_parameters["path"][i] - self.internal_parameters["centroid"] 
                    temp = np.dot(temp, temp)
                    if temp < distance:
                        distance = temp
                        self.internal_parameters["pp_id"] = i
                print("start id:", self.internal_parameters["pp_id"])
            check_id = (self.internal_parameters["pp_id"] + 1) % n_points
            temp = self.internal_parameters["path"][check_id] - self.internal_parameters["centroid"] 
            temp = np.dot(temp, temp)
            if temp < self.internal_parameters["L2"]: 
                self.internal_parameters["pp_id"] = check_id 
                print("new direction id:", check_id, "L2=", self.internal_parameters["L2"])
                print("centroid:", self.internal_parameters["centroid"], "direction:", self.internal_parameters["path"][self.internal_parameters["pp_id"]])
            self.controlled_parameters["direction"] = self.internal_parameters["path"][self.internal_parameters["pp_id"]]
        self.dmd_image = self.controlled_parameters["path_set"]*self.create_directed_pattern(self.controlled_parameters["shape"])
    
    def create_directed_pattern(self, width):
        centroid = self.internal_parameters["centroid"]
        self.internal_parameters["segment_id"] = self.processed_image[centroid[0],centroid[1]]
        if self.controlled_parameters["id_set"]:
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = self.processed_image[id_x,id_y]
            self.controlled_parameters["id_set"] = False
        self.internal_parameters["L2"] = (0.75/np.pi)*np.sum((self.processed_image==self.internal_parameters["segment_id"]))
        m = measure.moments(self.processed_image==self.internal_parameters["segment_id"])
        self.internal_parameters["centroid"] = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
        projection = self.controlled_parameters["direction"] - self.internal_parameters["centroid"]
        #print(self.internal_parameters["segment_id"], self.internal_parameters["centroid"])
        return cell_edge(self.processed_image,25)*sector_mask(self.processed_image.shape,self.internal_parameters["centroid"], projection, width)*(self.processed_image==self.internal_parameters["segment_id"])

    def create_path(self):
        '''
        Comment 
        '''
        '''
        center = np.array([500,600])
        radius = 350 
        n_points = 40
        angles = np.arange(0,2*np.pi,2*np.pi/n_points)
        assert angles.shape[0] == n_points
        path = np.ones((n_points,2))
        path[:,0] = center[0] + radius * np.cos(angles)
        path[:,1] = center[1] + radius * np.sin(angles)
        '''
        center = np.array([500,600])
        radius = 270 
        n_points = 40
        path = np.ones((n_points,2))
        vertices = np.tile(center,(4,1)) + np.array([[radius,radius],[-radius,radius],[-radius,-radius],[radius,-radius]])
        for i in range(4):
            for j in range(int(n_points/4)):
                index = i*int(n_points/4) + j
                path[index] = vertices[i] + 4 * j * (vertices[((i+1) % 4)] - vertices[i]) / n_points
        return path     

    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class ControlLEXY():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction and changes direction after a set number of frames
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((684,608))
        self.on = False
        self.frame_count = 0
        self.led_power = 1
        self.intensity = 1
        #self.led_set = False
        self.file = None

        # controlled parameters
        self.controlled_parameters = {
            "id": np.array([0,0]),
            "id_set": False,
            "led_set": False,
            "control_strength": 1.0
        }
        self.parameter_controls = {
            "double_left": "id",
            "double_left_set": "id_set",
            "textbox": "control_strength",
            "right": None,
            "right_set": "led_set"
        }

        # internal parameters
        self.internal_parameters = {
            "segment_id": 1,
            "nucleus": np.zeros(self.processed_image.shape),
            "cytosol": np.zeros(self.processed_image.shape),
            "bg": 1.0,
            "ratio": 1.0,
            "led_power": 1,
            "LEXY_parameters": (0.5,0.5,0.5/2.7,15) #(a1, a2, b, imaging interval[s])
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.controlled_parameters["id_set"]:
            temp = segment_cells2(cam_image)
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = temp[id_x,id_y]
            self.internal_parameters["nucleus"] = 1*(temp == self.internal_parameters["segment_id"])

            temp_edge = cell_edge(temp, 7)
            #print("temp edge", np.sum(temp), np.sum(temp_edge))
            temp = temp * (temp_edge == False)
            #print("temp edge", np.sum(temp), np.sum(temp_edge))
            self.dmd_image = (temp == self.internal_parameters["segment_id"])
            #print("dmd", np.sum(self.dmd_image))

            self.internal_parameters["cytosol"] = segment_cells3(cam_image)
            self.internal_parameters["cytosol"] = (self.internal_parameters["cytosol"] == self.internal_parameters["segment_id"])
            self.internal_parameters["cytosol"] = 1*self.internal_parameters["cytosol"] - 1* self.internal_parameters["nucleus"]
            self.processed_image = 1*(self.internal_parameters["cytosol"] != 0)

            bg_thresholds = filters.threshold_multiotsu(cam_image, classes=4)
            bg_mask = cam_image < bg_thresholds[0]
            #bg_mask = filters.gaussian(bg_mask,  sigma=2)
            #bg_mask = bg_mask > 0 
            self.processed_image += 2*bg_mask 
            self.internal_parameters["bg"] = cam_image[bg_mask!=0].mean()
            print("bg:", self.internal_parameters["bg"])
            print("cytosol:", cam_image[self.internal_parameters["cytosol"]!=0].mean())

            self.controlled_parameters["id_set"] = False

        if self.controlled_parameters["led_set"]:
            self.on = True
            #self.led_set = True 
            c_i = np.mean(cam_image[self.dmd_image != 0])
            c_o = np.mean(cam_image[(self.internal_parameters["cytosol"]*cell_edge(self.internal_parameters["nucleus"],15)) != 0])
            self.internal_parameters["ratio"] = (c_i - self.internal_parameters["bg"]) / (c_o - self.internal_parameters["bg"])
            #self.internal_parameters["ratio"] = (cam_image[self.internal_parameters["nucleus"]!=0].mean() - self.internal_parameters["bg"]) / (cam_image[self.internal_parameters["cytosol"]!=0].mean() - self.internal_parameters["bg"])
            #print(np.sum(self.dmd_image), np.sum(self.internal_parameters["cytosol"]), "in ratio computation")
            #print(self.internal_parameters["ratio"], cam_image[self.internal_parameters["nucleus"]!=0].mean(), cam_image[self.internal_parameters["cytosol"]!=0].mean())

    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        
        if self.controlled_parameters["led_set"]:
            a1, a2, b, T = self.internal_parameters["LEXY_parameters"]
            e = self.internal_parameters["ratio"] - self.controlled_parameters["control_strength"]
            u = (-a2*self.internal_parameters["ratio"]**2 + (b-a2)*self.internal_parameters["ratio"] + b + 0.5*10.0*e) / (a1*(self.internal_parameters["ratio"]**2 + self.internal_parameters["ratio"]))
            if u > 1: u = 1
            if u < 0: u = 0

            self.internal_parameters["led_power"] = u*T*1000
            self.intensity = int(u*255)

        self.led_power = int(self.internal_parameters["led_power"])
        if not self.on: 
            self.led_power = 0
            self.intensity = 0
        #self.dmd_image = self.on * self.dmd_image

    def write_data(self, time, date):
        if self.file is None:
            self.file = open("Acq_"+str(date)+"_"+time[11:13]+"-"+time[14:16]+".txt", "w")
        self.file.write(f'{self.internal_parameters["ratio"]} {self.internal_parameters["led_power"]} {self.controlled_parameters["control_strength"]} \n')
        print("ratio: ", self.internal_parameters["ratio"], "power: ", self.internal_parameters["led_power"])
    
    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        queue_dict: blabalabl
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


class PID_LEXY():
    def __init__(self):
        '''
        Model that automatically illuminates the cell edge in a set direction and changes direction after a set number of frames
        '''
        # processed image
        self.processed_image = np.zeros((1030,1300))
        self.dmd_image = np.zeros((684,608))
        self.on = False
        self.frame_count = 0
        self.led_power = 1
        self.intensity = 1
        #self.led_set = False
        self.file = None

        # controlled parameters
        self.controlled_parameters = {
            "id": np.array([0,0]),
            "id_set": False,
            "led_set": False,
            "control_strength": 1.0
        }
        self.parameter_controls = {
            "double_left": "id",
            "double_left_set": "id_set",
            "textbox": "control_strength",
            "right": None,
            "right_set": "led_set"
        }

        # internal parameters
        self.internal_parameters = {
            "segment_id": 1,
            "nucleus": np.zeros(self.processed_image.shape),
            "cytosol": np.zeros(self.processed_image.shape),
            "bg": 1.0,
            "ratio": 1.0,
            "led_power": 1,
            "LEXY_parameters": (0.5,0.5,0.5/2.7,15), #(a1, a2, b, imaging interval[s])
            "int_error": 0.0
        }

    def process_step(self, cam_image):
        '''
        Image processing of the camera image.
        no return value, just changes the internal variable processed_image
        '''
        if self.controlled_parameters["id_set"]:
            temp = segment_cells2(cam_image)
            id_x, id_y= self.controlled_parameters["id"]
            self.internal_parameters["segment_id"] = temp[id_x,id_y]
            self.internal_parameters["nucleus"] = 1*(temp == self.internal_parameters["segment_id"])

            temp_edge = cell_edge(temp, 7)
            #print("temp edge", np.sum(temp), np.sum(temp_edge))
            temp = temp * (temp_edge == False)
            #print("temp edge", np.sum(temp), np.sum(temp_edge))
            self.dmd_image = (temp == self.internal_parameters["segment_id"])
            #print("dmd", np.sum(self.dmd_image))

            self.internal_parameters["cytosol"] = segment_cells3(cam_image)
            self.internal_parameters["cytosol"] = (self.internal_parameters["cytosol"] == self.internal_parameters["segment_id"])
            self.internal_parameters["cytosol"] = 1*self.internal_parameters["cytosol"] - 1* self.internal_parameters["nucleus"]
            self.processed_image = 1*(self.internal_parameters["cytosol"] != 0)

            bg_thresholds = filters.threshold_multiotsu(cam_image, classes=4)
            bg_mask = cam_image < bg_thresholds[0]
            #bg_mask = filters.gaussian(bg_mask,  sigma=2)
            #bg_mask = bg_mask > 0 
            self.processed_image += 2*bg_mask 
            self.internal_parameters["bg"] = np.median(cam_image[bg_mask!=0]) #.mean()
            print("bg:", self.internal_parameters["bg"])
            print("cytosol:", cam_image[self.internal_parameters["cytosol"]!=0].mean())

            self.controlled_parameters["id_set"] = False

        if self.controlled_parameters["led_set"]:
            self.on = True
            #self.led_set = True 
            c_i = np.mean(cam_image[self.dmd_image != 0])
            c_o = np.mean(cam_image[(self.internal_parameters["cytosol"]*cell_edge(self.internal_parameters["nucleus"],15)) != 0])
            self.internal_parameters["ratio"] = (c_i - self.internal_parameters["bg"]) / (c_o - self.internal_parameters["bg"])
            #self.internal_parameters["ratio"] = (np.median(cam_image[self.internal_parameters["nucleus"]!=0]) - self.internal_parameters["bg"]) / (np.median(cam_image[self.internal_parameters["cytosol"]!=0]) - self.internal_parameters["bg"])
            #print(np.sum(self.dmd_image), np.sum(self.internal_parameters["cytosol"]), "in ratio computation")
            #print(self.internal_parameters["ratio"], cam_image[self.internal_parameters["nucleus"]!=0].mean(), cam_image[self.internal_parameters["cytosol"]!=0].mean())

    def controler_step(self):
        '''
        This function wraps all the controll processing and in the end sets the new dmd_image
        '''
        
        if self.controlled_parameters["led_set"]:
            a1, a2, b, T = self.internal_parameters["LEXY_parameters"]
            e = self.internal_parameters["ratio"] - self.controlled_parameters["control_strength"]
            self.internal_parameters["int_error"] += e
            #u = (-a2*self.internal_parameters["ratio"]**2 + (b-a2)*self.internal_parameters["ratio"] + b + 0.5*self.controlled_parameters["control_strength"]*e) / (a1*(self.internal_parameters["ratio"]**2 + self.internal_parameters["ratio"]))
            u = 0.9 * (e + (0.56/0.45) * 0.1 * self.internal_parameters["int_error"])
            if u > 1: u = 1
            if u < 0: u = 0

            self.internal_parameters["led_power"] = u*T*1000
            self.intensity = int(u*255)

        self.led_power = int(self.internal_parameters["led_power"])
        if not self.on: 
            self.led_power = 0
            self.intensity = 0
        #self.dmd_image = self.on * self.dmd_image

    def write_data(self, time, date):
        if self.file is None:
            self.file = open("Acq_"+str(date)+"_"+time[11:13]+"-"+time[14:16]+".txt", "w")
        self.file.write(f'{self.internal_parameters["ratio"]} {self.intensity} {self.controlled_parameters["control_strength"]} \n')
        print("ratio: ", self.internal_parameters["ratio"], "power: ", self.internal_parameters["led_power"])
    
    def set_parameters(self, queue_dict):
        '''
        This function takes a dictionary containing the changed parameter values
        It assigns those to the entries in controlled parameters with the same key
        '''
        for control in queue_dict:
            try:
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass


