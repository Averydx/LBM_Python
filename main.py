import pygame as pg
from LBM_class import LBM
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm   

def distance(x1,y1,x2,y2): 
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def main(): 
    lbm = LBM(400,400,0.6,(200,200))
    # while(True): 
    #     plt.imshow(lbm.solve())
    #     plt.pause(.01)
    #     plt.cla()

# initialize pygame
    pg.init()
    screen_size = (400, 400)
    
    # create a window
    screen = pg.display.set_mode(screen_size)
    pg.display.set_caption("LBM")
    
    # clock is used to set a max fps
    clock = pg.time.Clock()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        mouse_pos = pg.mouse.get_pos();
        lbm.mouse_pos = mouse_pos

        #update screen 
        speeds =lbm.solve()
        im = Image.fromarray(np.uint8(cm.bwr(speeds)*255))
        mode = im.mode
        size = im.size
        data = im.tobytes()
  
        # Convert PIL image to pygame surface image
        py_image = pg.image.fromstring(data, size, mode)

        screen.blit(py_image,(0,0))


        # flip() updates the screen to make our changes visible
        pg.display.flip()
        
        # how many updates per second
        clock.tick(60)

if __name__ == "__main__": 
    main()