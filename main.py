from feedback_model import *
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events
import numpy as np
import matplotlib.pyplot as plt
from tkinter import * 
from tkinter import simpledialog
from PIL import Image, ImageTk
import time
import copy
import datetime
import threading
import queue
from skimage import filters, feature, transform, measure, segmentation, io
import scipy
from functions import *
'''
LAYOUT:
ControlThread runnning a loop with all MM core commands
ModelThread running the image processing and control
GUI in main thread

TO DO:
multi channel acquisition
'''


'''
Initializations and global variables
    demo: determines if demo mode is used 
    dt: time interval of image update when no acquisition is running in ms, cannot really go lower than 1700
    max_height and offset: set image display size
'''
demo = False 
overnight = False
dt = 1700
max_height = 900
offset = 0
cal_file_path = "cal_file_60.txt"


'''
Hardware "setup"
    gets MM core object and its useful parameters
'''
bridge = Bridge()
print(bridge.get_core())
core = bridge.get_core()
if not(demo):
    DMD = core.get_slm_device()
    DMD_type = core.get_device_type(DMD)
    dmd_height, dmd_width = core.get_slm_height(DMD), core.get_slm_width(DMD)
    im_height, im_width = core.get_image_height(), core.get_image_width()
else:
    dmd_height,dmd_width = 600, 680
    demo_stream = DemoImageStream(r"C:\Users\49177\Pictures\acq\2021-09-15rac onoff_1\Full resolution\2021-09-15rac onoff_NDTiffStack.tif")
    im_height, im_width = demo_stream.im_height, demo_stream.im_width


class Calibration():
    def __init__(self):
        self.slm_points, self.cam_points = self.read_points()
        self.t_slmtocam = np.linalg.inv(transform.estimate_transform("affine", self.slm_points,self.cam_points).params)
        self.t_camtoslm = np.linalg.inv(transform.estimate_transform("affine", self.cam_points,self.slm_points).params)

    def read_points(self):
        try:
            cal_file = open(cal_file_path,"r")
            slm_points_r, cam_points_r = [],[]
            for line in cal_file:
                if line[0] == "#":
                    points = line.split(sep=" ")
                    slm_points_r.append([int(points[1]),int(points[2])])
                else:
                    points = line.split(sep=" ")
                    cam_points_r.append([int(points[0]),int(points[1])])
            slm_points_r = np.array(slm_points_r)
            cam_points_r = np.array(cam_points_r)
            cal_file.close()
        except:
            print("No file found")
            slm_points_r, cam_points_r = np.zeros((3,2)),np.zeros((3,2))
            #temp = simpledialog.askstring("cal_points","Enter points", parent= root)
            #slm_points_r = np.array(eval(temp))
            #print(slm_points_r)
        return slm_points_r, cam_points_r
    
    def write_points(self):
        cal_file = open(cal_file_path, "w")
        for line in self.slm_points.tolist():
            cal_file.write(f"# {line[0]} {line[1]} \n")
        for line in self.cam_points.tolist():
            cal_file.write(f"{line[0]} {line[1]} \n")
        cal_file.close()
        print(self.cam_points, self.slm_points)

    def set_calibration_image(self):
        f = np.vectorize(gaussian_2d,signature='(),(),(k),()->(m,n)',excluded=['0','1','sigma'])
        if np.allclose(self.slm_points, np.zeros((3,2))):
            temp = simpledialog.askstring("cal_points","Enter points", parent= root)
            self.slm_points = np.array(eval(temp)) 
        self.slm_calibration_image = np.sum(f(dmd_width,dmd_height,np.flip(self.slm_points,axis=1),1),axis=0)
        print(np.flip(self.slm_points,axis=1))
        self.slm_calibration_image = image_to_uint8(self.slm_calibration_image)
        self.slm_calibration_image = self.slm_calibration_image.flatten()
        cal_event.set()
        dmd_control_queue.put({"image":self.slm_calibration_image, "power": 1000, "set": False})
    
    def get_calibration_points(self):
        self.cam_calibration_image = image_control_queue.get(block=True)
        cal_event.clear()
        self.cam_calibration_image = filters.gaussian(self.cam_calibration_image,sigma=3)
        temp_points = feature.peak_local_max(self.cam_calibration_image, num_peaks=3, exclude_border=False)
        self.cam_points = temp_points
        self.cam_points = self.cam_points[np.argsort(self.cam_points[:,0])]
        self.slm_points = self.slm_points[np.argsort(self.slm_points[:,0])]
        self.t_slmtocam = np.linalg.inv(transform.estimate_transform("affine", self.slm_points,self.cam_points).params)
        self.t_camtoslm = np.linalg.inv(transform.estimate_transform("affine", self.cam_points,self.slm_points).params)
        self.write_points()

    def transform_slmtocam(self, slm_pixels):
        cam_pixels = np.zeros((im_height,im_width))
        cam_pixels[:slm_pixels.shape[0],:slm_pixels.shape[1]] += slm_pixels
        cam_pixels = scipy.ndimage.affine_transform(cam_pixels, self.t_slmtocam)
        if np.max(cam_pixels)>0: cam_pixels *= 255/np.max(cam_pixels)
        cam_pixels = cam_pixels.astype("uint8")
        return cam_pixels
        
    def transform_camtoslm(self, cam_pixels):
        slm_pixels = np.zeros((dmd_height,dmd_width))
        temp_pixels = scipy.ndimage.affine_transform(cam_pixels, self.t_camtoslm)
        slm_pixels += temp_pixels[:dmd_height,:dmd_width]
        if np.max(slm_pixels)>0: slm_pixels *= 255/np.max(slm_pixels)
        slm_pixels = slm_pixels.astype("uint8")
        return slm_pixels


'''
User interface
'''
class mainWindow():
    def __init__(self,parent):
        self.frame = Frame(parent, height=min(max_height,im_height)+100, width=im_width)
        self.frame.pack()
        self.canvas = Canvas(self.frame, height=min(max_height,im_height), width=im_width)
        self.canvas.place(x=-2,y=-2)

        self.pixels = np.zeros((min(max_height,im_height),im_width))
        self.processed_image = np.zeros((min(max_height,im_height),im_width))
        self.dmd_pattern = np.zeros((im_height,im_width))
        self.display_it = 0
        self.im=Image.fromarray(self.pixels)
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.display_image = self.canvas.create_image(0,0,image=self.photo,anchor=NW)
        
        self.canvas.bind("<Double-Button 1>",self.double_left)
        self.canvas.bind("<Button 3>", self.right)
        self.canvas.bind("<Button 2>", self.display_switch)
        self.canvas.bind("<Button 1>", self.left)
        self.canvas.bind("<ButtonRelease 1>", self.left_release)
        
        self.ill_size = 1.0
        self.ill_size_box = Text(root, height=2, width=5)
        self.ill_size_box.place(x=900,y=self.pixels.shape[0]+1)
        self.ill_size_box.insert(END, '1.5')

        self.set_calibrate_button = Button(root,height=2, width=10, text="Set Calibration Image", command=calibration.set_calibration_image)
        self.set_calibrate_button.place(x=1, y= self.pixels.shape[0]+30)
        self.get_calibrate_button = Button(root,height=2, width=10, text="Get Calibration", command=calibration.get_calibration_points)
        self.get_calibrate_button.place(x=100, y= self.pixels.shape[0]+30)
        self.ill_shape_button = Button(root,height=2, width=10, text="Ill shape", command=self.textbox)
        self.ill_shape_button.place(x=900, y= self.pixels.shape[0]+30)

        self.acq_args_box = Text(root, height=1, width=60)
        self.acq_args_box.place(x= 200, y =self.pixels.shape[0]+1)
        self.acq_args_box.insert(END, '{"num_time_points":400, "time_interval_s":15, "channel_group": "Channel", "channels":["4-mCherry"], "channel_exposures_ms":[400]}')
        self.im_c = 0
        self.acq_path_box = Text(root, height=1, width=20)
        self.acq_path_box.place(x= 200, y =self.pixels.shape[0]+30)
        self.acq_path_box.insert(END, 'F:\\acq\\')
        self.acq_name_box = Text(root, height=1, width=20)
        self.acq_name_box.place(x= 200, y =self.pixels.shape[0]+60)
        self.acq_name_box.insert(END, 'acquisition name')
        self.acq_button = Button(root,height=2, width=10, text="Acquire", command=self.acquire)
        self.acq_button.place(x=500, y= self.pixels.shape[0]+30)
        self.test = Image.fromarray((255*circle(im_width,max_height,[int(im_width/2),int(max_height/2)],100)).astype("uint8"))
        self.refresh()

    def refresh(self):
        if not image_ui_queue.empty():
            self.pixels = image_ui_queue.get()
            self.processed_image = processed_ui_queue.get()
        if not dmd_ui_queue.empty():
            self.dmd_pattern = dmd_ui_queue.get()
        if self.display_it==0: self.im=Image.fromarray(self.pixels)
        elif self.display_it==1: self.im=Image.fromarray(self.dmd_pattern)
        else: self.im=Image.fromarray(self.processed_image)
        self.photo = ImageTk.PhotoImage(self.im)
        self.canvas.itemconfig(self.display_image, image=self.photo)
        self.canvas.after(int(dt/2), self.refresh)

    def double_left(self, event):
        parameter_queue.put({"double_left": np.array([event.y,event.x]), "double_left_set": True})
    
    def right(self, event):
        parameter_queue.put({"right":[event.y,event.x], "right_set": True})
    
    def left(self, event):
        parameter_queue.put({"left":[event.y,event.x], "left_set": True})
        
    def left_release(self, event):
        parameter_queue.put({"left_release":[event.y,event.x], "left_release_set": True})

    def display_switch(self, event):
        self.display_it = (self.display_it + 1) % 3
    
    def textbox(self):
        parameter_queue.put({"textbox": float(self.ill_size_box.get("1.0","end-1c"))})

    def acquire(self):
        self.acq_args = eval(self.acq_args_box.get("1.0","end-1c"))
        self.acq_path = self.acq_path_box.get("1.0","end-1c")
        self.acq_name = self.acq_name_box.get("1.0","end-1c")
        self.acq_info = [self.acq_args,self.acq_path,self.acq_name]
        acq_queue.put(self.acq_info)
        acq_event.set()


'''
events, queues
'''
main_event = threading.Event()
acq_event = threading.Event()
cal_event = threading.Event()
image_control_queue = queue.LifoQueue()
acq_queue = queue.Queue()
dmd_control_queue = queue.Queue()
parameter_queue = queue.Queue()
dmd_ui_queue = queue.Queue()
dmd_acq_queue = queue.Queue()
image_ui_queue = queue.LifoQueue()
processed_ui_queue = queue.LifoQueue()
id_queue = queue.Queue()


'''
Threads
'''
class ModelThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cam_image = np.zeros((im_height,im_width))
        #self.model = ControlLEXY()
        self.model = AutomaticPath()
        #self.model = ManualRepositioning()
        #self.model = Test()
        self.timer = time.time()

    def run(self):
        while not main_event.isSet():
            if not image_control_queue.empty():
                self.cam_image = image_control_queue.get()
            #image acquisition
                self.model.process_step(self.cam_image)
                image_ui_queue.put(image_to_uint8(self.cam_image))
                processed_ui_queue.put(image_to_uint8(self.model.processed_image))
            #image processing
            while not parameter_queue.empty():
                self.model.set_parameters(parameter_queue.get())
            if dmd_control_queue.empty():
                self.model.controler_step()
                try:
                    self.queue_dmd_image(self.model.dmd_image, self.model.led_power, self.model.led_set)
                except:
                    try: 
                        self.queue_dmd_image(self.model.dmd_image, 1000, False, self.model.intensity)
                    except:
                        self.queue_dmd_image(self.model.dmd_image, 1000, False)
            #control processing
            #dmd control
                if acq_event.isSet():
                    try: self.model.write_data(time.ctime(), datetime.date.today())
                    except: pass
                else:
                    try:
                        if self.model.file is not None:
                            self.model.file.close()
                            self.model.file = None
                    except: pass 
            elapsed = time.time() - self.timer
            self.timer = time.time()
            #if elapsed > 0.01: print("model time:", elapsed)

    def queue_dmd_image(self,dmd_image, power, set, intensity=None):
        dmd_image = image_to_uint8(dmd_image, max_i=intensity)
        dmd_ui_queue.put(dmd_image)
        if acq_event.isSet(): dmd_acq_queue.put(dmd_image)
        dmd_image = calibration.transform_camtoslm(dmd_image)
        dmd_image = image_to_uint8(dmd_image, max_i=intensity)
        #dmd_image = np.flip(dmd_image,1)
        dmd_image = dmd_image.flatten()
        if not cal_event.is_set(): dmd_control_queue.put({"image":dmd_image, "power": power, "set": set})
        #if acq_event.isSet(): dmd_acq_queue.put(dmd_image)
          

class ControlThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.timer = time.time()
    
    def run(self):
        while not main_event.isSet():
            while not main_event.isSet() and not acq_event.isSet():
                    if not demo:
                        self.put_image_in_queue(self.snap_from_core())
                    else:
                        self.put_image_in_queue(demo_stream.get())
                    if not dmd_control_queue.empty() and not demo:
                        dmd_dict = dmd_control_queue.get()
                        core.set_slm_image(DMD,dmd_dict["image"])
                        if dmd_dict["set"]: 
                            #print("asd")
                            core.set_property("Mightex_BLS(USB)", "normal_CurrentSet", 1)
                            time.sleep(dmd_dict["power"]*0.001)
                            core.set_property("Mightex_BLS(USB)", "normal_CurrentSet", 0)
                        #print(core.get_config_group_state("LED power").get_verbose(), "asd")
                    if demo:
                        dump = dmd_control_queue.get()

                    time.sleep(0.001*(dt - int(1000*time.time()) % dt))
                    elapsed = time.time() - self.timer
                    #print(elapsed)
                    self.timer = time.time()
                    #if elapsed > 0.01: print("control time:", elapsed)
            if not main_event.isSet():
                self.run_acquisition()
    
    def snap_from_core(self):
        core.snap_image()
        tagged_image = core.get_tagged_image()
        return np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
    
    def put_image_in_queue(self, image):
        #image = image_to_uint8(image)
        #image = image[offset:offset+max_height,:]
        image_control_queue.put(image)
    
    def run_acquisition(self):
        self.acq_info = acq_queue.get()
        self.acq_args = self.acq_info[0]
        self.acq_path = self.acq_info[1]
        self.acq_name = self.acq_info[2]
        with Acquisition(directory=str(self.acq_path), name=str(datetime.date.today())+self.acq_name, show_display=True, image_process_fn=self.acq_process, pre_hardware_hook_fn=self.dmd_hook) as acq:
            events = multi_d_acquisition_events(**(self.acq_args))
            acq.acquire(events)
        acq_event.clear()
        if overnight:
            main_event.set()
            core.set_config("LED", "Blue OFF")
    
    def acq_process(self,image,metadata):
        if not demo:
            self.put_image_in_queue(image)
        else:
            self.put_image_in_queue(demo_stream.get())
        if not dmd_acq_queue.empty():
            dmd_image = dmd_acq_queue.get()
        else:
            dmd_image = np.zeros((im_height,im_width))
        #dmd_image = np.reshape(dmd_image, newshape=(dmd_height,dmd_width))
        #dmd_image = calibration.transform_slmtocam(dmd_image)
        dmd_image = dmd_image.astype("uint16")
        dmd_md = copy.deepcopy(metadata)
        dmd_md["Channel"] = "DMD"
        #print("imgdtype:", image.dtype)
        return [(image, metadata),(dmd_image,dmd_md)]
    
    def dmd_hook(self, event):
        if not dmd_control_queue.empty() and not demo:
            dmd_dict = dmd_control_queue.get()
            core.set_slm_image(DMD,dmd_dict["image"])
            if dmd_dict["set"]: 
                core.set_property("Mightex_BLS(USB)", "normal_CurrentSet", 1)
                time.sleep(dmd_dict["power"]*0.001)
                core.set_property("Mightex_BLS(USB)", "normal_CurrentSet", 0)
        return event
            
    
'''
Actual execution
'''
calibration = Calibration()
control_thread = ControlThread()
control_thread.start()
model_thread = ModelThread()
model_thread.start()
root = Tk()
window = mainWindow(root)
root.mainloop()
main_event.set()
model_thread.join()
control_thread.join()
threading._shutdown()
