# I wish I had the .blend assets to automate data creation

""" 
Rapidly take screenshots for gathering data 
""" 
from PIL import ImageGrab 
from datetime import datetime 
from time import sleep 

# Choose which cars and maps will be recorded 
cars = ["dominus", "fennec", "octane"] 
maps = ["mannfield_day", "mannfield_night", "wasteland", "neo_tokyo", "champions_field_day", "champions_field_night", "utopia_coliseum_day", "utopia_coliseum_night"] 

max_ss = 2000    # number of screenshots to take 
ss_count = 0     # counter for loop 

for car in cars: 
    print(f'PREPARE FOR {car}') 
    for map_ in maps: 
        print(f'Start a {car} on {map_} and press the red E key') 
        # Wait to allow for alt+tab into game 
        sleep(5) 
        while ss_count < max_ss: 
            im = ImageGrab.grab() 
            dt = datetime.now() 
            # Note: changed next line from r-string to f-string
            file_path_prefix = f"..\\data\\img\\screenshots\\{map_}\\{car}\\" 
            fname = "{}{}_{}.jpg".format(file_path_prefix, car, ss_count) 
            im.save(fname, 'jpeg') 
            print(f'Saved {car} # {ss_count}') 
            ss_count += 1