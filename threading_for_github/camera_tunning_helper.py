import cv2
import subprocess

def camera_current_setting(k, cap):

    if k == ord('g'):
        print("******************************")
        print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
        print("Brightness = ",cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print("Contrast = ",cap.get(cv2.CAP_PROP_CONTRAST))
        print("Saturation = ",cap.get(cv2.CAP_PROP_SATURATION))
        print("Gain = ",cap.get(cv2.CAP_PROP_GAIN))
        print("Hue = ",cap.get(cv2.CAP_PROP_HUE))
        print("Exposure = ",cap.get(cv2.CAP_PROP_EXPOSURE))
        print("zoom = ",cap.get(cv2.CAP_PROP_ZOOM))
        print("Focus = ",cap.get(cv2.CAP_PROP_FOCUS))
        print("Auto Focus = ",cap.get(cv2.CAP_PROP_AUTOFOCUS))
        print("Auto Exposure = ",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print("******************************")  

def contrast_ctrl(Contrast, k, cap):
    

    if k == ord('y'):
        if(Contrast <= 0):
            Contrast = 0
        else:
            Contrast-=1


        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line
        
        contra = cap.set(cv2.CAP_PROP_CONTRAST, Contrast)
        
        if contra:
            print(cap.get(cv2.CAP_PROP_CONTRAST))
        else:
            print("contra --> False")
    elif k == ord('h'):
        if(Contrast >= 64):
            Contrast = 64
        else:
            Contrast+=1

        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line
        
        contra = cap.set(cv2.CAP_PROP_CONTRAST, Contrast)
        
        if contra:
            print(cap.get(cv2.CAP_PROP_CONTRAST))
        else:
            print("contra --> False") 

    return Contrast    


def brightness_ctrl(Brightness, k, cap):
    

    if k == ord('e'):
        if(Brightness <= 0):
            Brightness = 0
        else:
            Brightness-=1

        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line
        
        bright = cap.set(cv2.CAP_PROP_BRIGHTNESS, Brightness)
        if bright:
            print(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        else:
            print("bright --> False")
    elif k == ord('d'):
        if(Brightness >= 64):
            Brightness = 64
        else:
            Brightness+=1
       
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line
        
        bright = cap.set(cv2.CAP_PROP_BRIGHTNESS, Brightness)
        if bright:
            print(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        else:
            print("bright --> False") 

    return Brightness    



def focus_ctrl(Focus, k, cap):

    if k == ord('r'):
        if(Focus <= 0):
            Focus = 0
        else:
            Focus-=1
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0) # disable auto focus for manual chnging
        cap.set(cv2.CAP_PROP_FOCUS, Focus)
        print(cap.get(cv2.CAP_PROP_FOCUS))
    elif k == ord('f'):
        if(Focus >= 127):
            Focus = 127
        else:
            Focus+=1
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0) # disable auto focus for manual chnging
        cap.set(cv2.CAP_PROP_FOCUS, Focus)
        print(cap.get(cv2.CAP_PROP_FOCUS))
        
    return Focus


def zoom_ctrl(zoom, k, cap):
    if k == ord('w'):
        if(zoom <= 0):
            zoom = 0
        else:
            zoom-=100
        cap.set(cv2.CAP_PROP_ZOOM,zoom)
        print(cap.get(cv2.CAP_PROP_ZOOM))
    elif k == ord('s'):
        if(zoom >= 16384):
            zoom = 16384
        else:
            zoom+=100
        cap.set(cv2.CAP_PROP_ZOOM,zoom)
        print(cap.get(cv2.CAP_PROP_ZOOM))  
    return zoom 


def exposure_ctrl(Exposure, k, cap):

    if k == ord('q'):
        if(Exposure <= 10):
            Exposure = 10
        else:
            Exposure-=10
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # disable auto exposure for manual changing
        # cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=3".split(' '))
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1".split(' '))
        subprocess.run(f"v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute={Exposure}".split(' '))
        print(cap.get(cv2.CAP_PROP_EXPOSURE))
    elif k == ord('a'):
        if(Exposure >= 1250):
            Exposure = 1250
        else:
            Exposure+=10
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # disable auto exposure for manual changing
        # cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=3".split(' '))
        subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1".split(' '))
        subprocess.run(f"v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute={Exposure}".split(' '))
        print(cap.get(cv2.CAP_PROP_EXPOSURE))
        
    return Exposure


def set_camera_settings(cap):

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0) # disable auto focus for manual chnging
    cap.set(cv2.CAP_PROP_FOCUS, 34)

    subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
    subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line
    contra = cap.set(cv2.CAP_PROP_CONTRAST, 25)

    subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=3".split(' '))
    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1".split(' '))
    # subprocess.run(f"v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute={560.0}".split(' '))
    
    cap.set(cv2.CAP_PROP_ZOOM, 9000)
    
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 32)

    cap.set(cv2.CAP_PROP_FPS, 30)


def auto_focus(focus_step, cap):
    
    Focus = 35
    focus_step += 1 
    target_focus = Focus-focus_step
    cap.set(cv2.CAP_PROP_FOCUS, target_focus)
    return focus_step , target_focus