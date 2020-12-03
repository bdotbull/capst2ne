import os
import cv2

def images_from_video(vid_dir, out_dir='', car='unspecified'):
    """**USE conda cv env** converts video to series of images.

    Args:
        vid_dir ([type]): [description]
        out_dir ([type]): [description]
        fps (int, optional): [description]. Defaults to 2.
    """
    # Read the video from specified path 
    cam = cv2.VideoCapture(vid_dir) 
    
    if not out_dir:
        out_dir = f'../data/img/screenshots/play/{car}/'

    # make new directory or throw error
    try: 
        if not os.path.exists(out_dir): 
            os.makedirs(f'{out_dir}') 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            name = f'{out_dir}frame' + str(currentframe) + '.jpg'
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

if __name__ == '__main__':
    print(cv2.__version__)
    
    test_dir = '/media/brandon/Elements/media/video/geforce_recordings/Rocket League/Rocket League 2020.11.22 - 22.00.56.16.DVR.mp4'

    images_from_video(test_dir, car='octane')