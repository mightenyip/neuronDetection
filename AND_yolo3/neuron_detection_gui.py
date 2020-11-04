import tkinter as tk
from tkinter import ttk, StringVar
from cv2 import cv2
from PIL import Image, ImageTk

class PBL_APP(tk.Tk): 
      
    # __init__ function for class tkinterApp  
    def __init__(self, *args, **kwargs):  
          
        # __init__ function for class Tk 
        tk.Tk.__init__(self, *args, **kwargs) 

        self.title("PBL Real-Time Neuron Detection")
        self.geometry("980x500")

          
        # creating a container 
        container = tk.Frame(self)   
        container.pack(side = "top", fill = "both", expand = True)  
   
        container.grid_rowconfigure(0, weight = 1) 
        container.grid_columnconfigure(0, weight = 1) 
   
        # initializing frames to an empty array 
        self.frames = {}   
   
        # iterating through a tuple consisting 
        # of the different page layouts 
        for F in (MainPage, SettingsPage): 
   
            frame = F(container, self) 
   
            # initializing frame of that object from 
            # MainPage, SettingsPage, page2 respectively with  
            # for loop 
            self.frames[F] = frame  
   
            frame.grid(row = 0, column = 0, sticky ="nsew") 
   
        self.show_page(MainPage) 
   
    # to display the current frame passed as 
    # parameter 
    def show_page(self, cont): 
        frame = self.frames[cont] 
        frame.tkraise() 

class MainPage(tk.Frame): 
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)

        self.camera_label = ttk.Label(self, text="CAMERA", font='Arial 14 bold')
        self.camera_label.grid(row = 0, column = 1, padx = 10, pady = 10)

        self.camera_mode = ttk.Label(self, text="Camera Mode")
        self.camera_mode.grid(row = 1, column = 1, padx = 10, pady = 10)


        self.camera_livestream = ttk.Button(self, text ="Livestream") 
        self.camera_livestream.grid(row = 1, column = 2, padx = 10, pady = 10)

        self.camera_upload = ttk.Button(self, text ="Upload") 
        self.camera_upload.grid(row = 1, column = 3, padx = 10, pady = 10)

        self.cell_detection = ttk.Label(self, text="Cell Detection")
        self.cell_detection.grid(row = 2, column = 1, padx = 10, pady = 10)


        self.on_cell_detection = ttk.Button(self, text ="ON", command=self.popup_cell_detection) 
        self.on_cell_detection.grid(row = 2, column = 2, padx = 10, pady = 10)

        self.off_cell_detection = ttk.Button(self, text ="OFF") 
        self.off_cell_detection.grid(row = 2, column = 3, padx = 10, pady = 10)

        self.capture_status = ttk.Label(self, text="Capture Status")
        self.capture_status.grid(row = 3, column = 1, padx = 10, pady = 10)

        self.start_capture = ttk.Button(self, text ="Start Capture", command=self.capture_start) 
        self.start_capture.grid(row = 3, column = 2, padx = 10, pady = 10)

        self.stop_capture = ttk.Button(self, text ="Stop Capture", command=self.capture_stop)
        self.stop_capture.grid(row = 3, column = 3, padx = 10, pady = 10)

        self.settings_label = ttk.Label(self, text="SETTINGS", font='Arial 14 bold')
        self.settings_label.grid(row = 4, column = 1, padx = 10, pady = 10)

        self.scale_label = ttk.Label(self, text="Scale")
        self.scale_label.grid(row = 5, column = 1, padx = 10, pady = 10)

        self.scale_var = StringVar(self)
        self.scale_var.set("Scale") # default value

        self.scale = ttk.OptionMenu(self, self.scale_var, "1:1", "1:2", "1:3", "1:4")
        self.scale.grid(row = 5, column = 2, padx = 10, pady = 10)

        self.duration_label = ttk.Label(self, text="Duration")
        self.duration_label.grid(row = 6, column = 1, padx = 10, pady = 10)

        self.duration_var = StringVar(self)
        self.duration_var.set("Duration") # default value

        self.duration = ttk.OptionMenu(self, self.duration_var, "20ms", "50ms", "100ms", "200ms")
        self.duration.grid(row = 6, column = 2, padx = 10, pady = 10)

        # self.time_label = ttk.Label(self, text="Time")
        # self.duration.grid(row = 7, column = 1, padx = 10, pady = 10)

        # self.timer_label = ttk.Label(self, text="00.00")



        self.imageFrame = tk.Frame(self, width=1280, height=1024)
        self.imageFrame.grid(row=0, column=4, rowspan = 7, padx=10, pady=2)

        self.lmain = tk.Label(self.imageFrame)
        self.lmain.grid(row=0, column=4)
        self.cap = cv2.VideoCapture(0)

        self.show_frame()
        
    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(10, self.show_frame)
    
    def capture_start(self):
        fps = 30
 
        # Width and height of the frames in the video stream
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        videoWriter = cv2.VideoWriter('MyOutput.avi', 
            cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
        
        success, frame = self.cap.read()
        
        # some variable
        numFramesRemaining = 10*fps - 1
        
        # loop until there are no more frames and variable > 0
        while success and numFramesRemaining > 0:
            videoWriter.write(frame)
            success, frame = self.cap.read()
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            numFramesRemaining -= 1
    
    def capture_stop(self):
        self.cap.release()
    
    def popup_cell_detection(self):
        win = tk.Toplevel()
        win.geometry("250x160")
        win.wm_title("Cell Detection Settings")

        version_label = ttk.Label(win, text="YOLO Version")
        version_label.grid(row = 0, column = 0, padx = 10, pady = 10)

        version_var = StringVar(win)
        version_var.set("YOLO V") # default value

        version = ttk.OptionMenu(win, version_var, "YOLOv3", "YOLOv4", "YOLOv5", "PP-YOLO")
        version.grid(row = 0, column = 1, padx = 10, pady = 10)

        detection_label = ttk.Label(win, text="Detection Type")
        detection_label.grid(row = 1, column = 0, padx = 10, pady = 10)

        detection_var = StringVar(win)
        detection_var.set("Detection") # default value

        detection = ttk.OptionMenu(win, detection_var, "Single Detection", "Tracking", "Multiple Detection")
        detection.grid(row = 1, column = 1, padx = 10, pady = 10)

        file_label = ttk.Label(win, text="File Name")
        file_label.grid(row = 2, column = 0, padx = 10, pady = 10)

        file_entry = tk.StringVar()
        filename_entry = ttk.Entry(win, width = 15, textvariable = file_entry)
        filename_entry.grid(row = 2, column = 1)


        save_button = ttk.Button(win, text="Save", command=win.destroy)
        save_button.grid(row=3, column=0)

        cancel_button = ttk.Button(win, text="Cancel", command=win.destroy)
        cancel_button.grid(row=3, column=1)


    


class SettingsPage(tk.Frame): 
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)


# Driver Code 
app = PBL_APP() 
app.mainloop() 
