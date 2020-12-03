import os
import cv2

def images_from_video(vid_dir, out_dir, fps=2):
    """**USE conda cv env** converts video to series of images.

    Args:
        vid_dir ([type]): [description]
        out_dir ([type]): [description]
        fps (int, optional): [description]. Defaults to 2.
    """
    # Read the video from specified path 
    cam = cv2.VideoCapture(vid_dir) 
    
    # make new directory or throw error
    try: 
        if not os.path.exists('data'): 
            os.makedirs('data') 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
    
            # writing the extracted images 
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 
