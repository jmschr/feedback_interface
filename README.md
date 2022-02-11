# feedback_interface
### Microscope software interface for optogenetic feedback using a DMD

This feedback interface consists of 3 files:

1. The main program which runs the image acquisition and dmd control. 
This is the file to be run inside any python IDE with Micromanager 2.x running and "Tools/Options/Run server on port 4827" enabled.

2. The feedback model file, which contains the classes that are used in the model thread of the main program.
These govern the image processing, mode of feedback and response to interactions with the interface

3. A functions file which just contains useful functions and classes so that they do not clutter the main program


Quick Use Guide:

- Calibration:
  - Click Set calibraion image (leftmost button), focus on the image in transmission channel (you can set this in MM)
  - Click Get calibration image (right next to it), wait
- Feedback model:
  - always contains a processed_image, dmd_image variable and process_step, controler_step and set_parameters functions
  - The controlled_parameters, internal_parameters and parameter_control dictionaries can be set as needed.
  - In parameter_controls you can just set any entry you do not need to None.
  - ! Assign the model in the ModelThread class of the main program to self.model
- Demo mode: 
  - in the "initializations" part of the program assign demo = True and in the "hardware setup" part input a tif stack in the file path for the demo_stream
  - This way you can test your feedback model without access to the microscope
- Display control:
  - by clicking the mouse wheel you can toggle between camera, dmd or processed image
