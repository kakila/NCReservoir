#------------------------------------------------------------
# Demo of the free motion of a symmetric top in the reference
# frame of the center of mass (fixed at the origin).
#
# Left-click the mouse to toggle the curve of the tip of the
#    x1 moving axis.
#
# Right-click to toggle display of the angular momentum and
#    angular velocity.
#
# Center click (=left+right click) to zoom.
# 
# Press ESC to end the program.
#
# by  E. Velasco, December 2004 and 
# by Federico Corradi 2014 federico@ini.phys.ethz.ch
#-------------------------------------------------------------
from visual import *
import numpy as np
from threading import Thread
import time as tt

class RetinaInputs():

    def __init__(self, setup, I1 = 1.0, I3 = 2.0,
                omega0=1.0, omega1=1.0, omega3 = 1.0,
                display_arms = False,
                ShowVectors = 0,
                ShowOrbits = True,
                L1 = 1.0,
                L3 = 3.0,
				BuildOffline = False):
        self.BuildOffline = BuildOffline
        # Define the global constants
        self.I1 = I1             # I1=I2 symmetric princ. moments of inertia
        self.omega0 = omega0          # positive by definition
        self.I3 = I3             # The tird principal moment of self
        self.omega3 = omega3
        self.display_arms = display_arms  #display cilider arms
        # Create the angular momentum (L) and angular velocity (w) vectors
        self.ShowVectors = ShowVectors  # Show L and w. 0=hide, 1=show
        self.ShowOrbits = ShowOrbits
        # Define some variables such as lenghts and time parametes
        self.L1 = L1
        self.L3 = L3
        self.LF = 0.7*L3   # lenght of the moving frame axis

        self.Omega_p = sqrt((I1*omega0)**2+(I3*self.omega3)**2)/I1  # precession omega
        self.Omega_e  = self.omega3*(I3-I1)/I1     # extra rotation around 3 

        self.t = 0
        self.Dt = 0.02/self.Omega_p
        self.PaintOrbit = False

        # Define the colors of all the objects
        self.CL = (1,0.4,1)          # Angular momentum
        self.COmega = color.green    # Angular velocity
        self.CMvFrm = color.yellow   # Moving coordinate system
        self.CObject = (0,0.6,0.9)   # Rotating top
        self.COrbit = (1,0.5,0.2)    # Orbit of the x1 axis

        self.CosTheta = I3*self.omega3/sqrt((I1*self.omega0)**2+(I3*self.omega3)**2)
        self.SinTheta = sqrt(1-self.CosTheta**2)
        self.cm = vector(0,0,0)

        self.setup = setup
        self.this_t = 0
        self.retina_channel = 2
        self.chip_channel = 0
        self.sync_dur = 50 #duration/framerate
        self.delay_after_sync = 2000
                
        self.raw_data = []
        self._init_everything()
        self.inputs_signals = []
        self.teach_signals = []
        self.delay_sync_time = 1000
        
        self.XMAX = 400 # x and y range over 0 to XMAX
        self.scene = scene
        self.scene.width = self.XMAX
        self.scene.fov = 0.01 # make effectively 2D
        self.scene.range = self.XMAX/2
        self.scene.center = (self.XMAX/2,self.XMAX/2)
        self.scene.height = self.XMAX+60 # titlebar plus toolbar 60 pixels high
        self.pixels = points()
        self.haspoints = True


    def MovingFrame(self):
        phi = self.Omega_p*self.t
        psi = pi/2-self.Omega_e*self.t
        CosPhi = cos(phi)
        SinPhi = sin(phi)
        CosPsi = cos(psi) 
        SinPsi = sin(psi)
        self.V1 = vector(CosPhi*CosPsi-self.CosTheta*SinPhi*SinPsi,SinPhi*CosPsi+self.CosTheta*CosPhi*SinPsi,self.SinTheta*SinPsi)
        self.V3 = vector(self.SinTheta*SinPhi,-self.SinTheta*CosPhi, self.CosTheta)
        self.V2 = cross(self.V3,self.V1)

    def Char_L(self, V):
        # A function that returns a "L" in the y-z plane displaced by V
        sz = 0.04 # size of letter
        return [V+vector(0,-sz,3*sz),V+vector(0,-sz,0), V+vector(0,sz,0)]

    def _init_everything(self):
        # Build a leter omega in the y-z plane
        sz_omega  = 0.04   # size of letter
        Npts = 10
        alpha0=2*pi/3
        Dalpha = (2*pi-alpha0)/(Npts-1)
        l_omega = []
        for k in range(Npts):
            alpha = alpha0+k*Dalpha
            l_omega.append(sz_omega*vector(0,cos(alpha)-1,1+sin(alpha)))
        for k in range(Npts):
            alpha = pi+k*Dalpha
            l_omega.append(sz_omega*vector(0,1+cos(alpha),1+sin(alpha)))
        self.letter_omega = array(l_omega) # Use a numeric array for efficiency

        # Properties of the display window 
        self.window = display(title="Retina Stimulus", y=750, x=850, width=600, height=490)#width=600, height=600)
        self.window.fullscreen = 0      # Change to 0 to get a floating window
        self.window.userspin = 0        # No rotation with mouse
        self.window.range = (4,5,4)
        self.window.forward =  (-1,0,0)
        self.window.up = (0,0,1)        # psoitive z axis vertically up!  
        self.window.ambient=ambient=0.1 # perfect for retina dvs (control mean firing rates)
        self.window.lights = [0.5*norm((1,-0.5,-0.2)),0.6*norm((1,0.5,0.2))]
        self.window.select()
        try:
            self.window.material=None
        except:
            pass

        Info = (self.I3/self.I1, self.omega3/self.omega0)
        label(text = "I3/I1 = %.2f    w3/w0 = %.2f" % Info, pos=(0,0,-3),
        height=20, color=color.red, box=0)

        # Define the unit vectors of the moving frame and the object at t=0
        self.MovingFrame()

        
        if self.ShowOrbits:
            self.MvFrm1 = curve(pos=[self.cm,self.LF*self.V1], color=self.CMvFrm, radius=0.02)
            self.MvTip1 = cone(pos=self.LF*self.V1, axis=0.1*self.LF*self.V1, radius=0.05, color=self.CMvFrm) 
            self.MvFrm2 = curve(pos=[self.cm,self.LF*self.V2], color=self.CMvFrm, radius=0.02)
            self.MvTip2 = cone(pos=self.LF*self.V2, axis=0.1*self.LF*self.V2, radius=0.05, color=self.CMvFrm) 
            self.MvFrm3 = curve(pos=[self.cm,self.LF*self.V3], color=self.CMvFrm, radius=0.02)
            self.MvTip3 = cone(pos=self.LF*self.V3, axis=0.1*self.LF*self.V3, radius=0.05, color=self.CMvFrm)

        self.tip = curve(color=self.COrbit)
        if self.ShowOrbits and self.PaintOrbit: tip.append(pos=self.MvTip1.pos+self.MvTip1.axis)

        #sync helix
        #self.sync = helix(pos=(2,3,2.6), axis=(-0.6, 0, -0.6), raidus=0.2) 
        #self.sync.set_radius(0.2)
        #self.sync.set_pos((2,3,2.6))
        #self.sync.set_axis((-0.6, 0, -0.6)) 

        # Define the object: a body (ellipsoid) +2 arms (cylinders)
        self.Body = ellipsoid(pos=self.cm, axis=self.L3*self.V3,
         height=self.L1, width=self.L1, color=self.CObject)
        if self.display_arms:
            self.Arm1 = cylinder(pos=-1.2*self.L1*self.V1, axis=2.4*self.L1*self.V1, radius=0.15*self.L1, color=self.CObject)
            self.Arm2 = cylinder(pos=-1.2*self.L1*self.V2, axis=2.4*self.L1*self.V2, radius=0.15*self.L1, color=self.CObject)

        self.L_body = curve(pos=[self.cm,(0,0,self.LF)], color=self.CL, radius=0.02,
        visible=self.ShowVectors)
        self.L_tip = cone(pos=(0,0,self.LF), axis=(0,0,0.1*self.LF), radius=0.05, color=self.CL,
        visible=self.ShowVectors)
        self.L_label = curve(pos=self.Char_L(self.L_tip.pos+1.2*self.L_tip.axis), radius= 0.015,
        color=self.CL, visible=self.ShowVectors)

        w = self.omega0*(cos(self.Omega_e*self.t)*self.V1+sin(self.Omega_e*self.t)*self.V2)+self.omega3*self.V3
        w = w/mag(w)*self.LF
        self.w_body = curve(pos=[self.cm,w], radius=0.02, color=self.COmega,
        visible=self.ShowVectors)
        self.w_tip = cone(pos=w, axis=0.1*w, radius=0.05, color=self.COmega,
        visible=self.ShowVectors)
        self.w_label = curve(pos=self.Char_omega(self.w_tip.pos+1.2*self.w_tip.axis), radius= 0.015,
        color=self.COmega, visible=self.ShowVectors) 

        # A function that returns a omega in the y-z plane displaced by V
    def Char_omega(self, V):
        char=array([V])+self.letter_omega  # Fast numeric array addition 
    	return char

    def start_position(self, I1 = 1.0, I3 = 2.0,
                omega0=1.0, omega1=1.0, omega3 = 1.0,
                display_arms = False,
                ShowVectors = 0,
                ShowOrbits = True,
                L1 = 1.0,
                L3 = 3.0):
        del self.inputs_signals
        del self.teach_signals
        del self.t
        del self.this_t
        del self.Dt 
        self.pixels.visible = False
        self.inputs_signals = []  
        self.teach_signals = []      
        # Define the global constants
        self.I1 = I1             # I1=I2 symmetric princ. moments of inertia
        self.I3 = I3             # The tird principal moment of inertia
        self.omega0 = omega0          # positive by definition
        self.omega3 = omega3
        self.display_arms = display_arms  #display cilider arms
        # Create the angular momentum (L) and angular velocity (w) vectors
        self.ShowVectors = ShowVectors  # Show L and w. 0=hide, 1=show
        self.ShowOrbits = ShowOrbits
        # Define some variables such as lenghts and time parametes
        self.L1 = L1
        self.L3 = L3
        self.LF = 0.7*L3   # lenght of the moving frame axis

        self.Omega_p = sqrt((I1*omega0)**2+(I3*self.omega3)**2)/I1  # precession omega
        self.Omega_e  = self.omega3*(I3-I1)/I1     # extra rotation around 3 

        self.t = 0
        self.this_t = 0
        self.Dt = 0.02/self.Omega_p
        self.PaintOrbit = False
        self.CosTheta = I3*self.omega3/sqrt((I1*self.omega0)**2+(I3*self.omega3)**2)
        self.SinTheta = sqrt(1-self.CosTheta**2)
        self.pixels = points()
        self.pixels.visible = True

    def plot_pix(self, nx, ny, nz, c):
        self.pixels.append(pos=(0,ny,nz), color=c)

    def _record_chip_activity(self, duration, neu_sync):
        out = self.setup.stimulate({}, send_reset_event=False, duration=duration) 
        #clean data
        raw_data = out[self.chip_channel].raw_data()
        sync_index = raw_data[:,1] == neu_sync
        start_time = np.min(raw_data[sync_index,0])
        out[self.chip_channel].t_start = start_time+self.delay_sync_time
        out[self.retina_channel].t_start = start_time+self.delay_sync_time
              
        self.out = out

    def go_sync(self):
        self.sync.set_y(np.random.random()+2.3)
        
    def update_ellipsoid(self):
        # Update the body and the moving frame
        self.Body.axis=self.L3*self.V3
        if self.display_arms:
            self.Arm1.pos=-1.2*self.L1*self.V1; 
            self.Arm1.axis=2.4*self.L1*self.V1
            self.Arm2.pos=-1.2*self.L1*self.V2; 
            self.Arm2.axis=2.4*self.L1*self.V2

        if self.ShowOrbits :
            self.MvFrm1.pos=[self.cm,self.LF*self.V1]
            self.MvTip1.pos=self.LF*self.V1; 
            self.MvTip1.axis=0.1*self.LF*self.V1
            self.MvFrm2.pos=[self.cm,self.LF*self.V2]
            self.MvTip2.pos=self.LF*self.V2; 
            self.MvTip2.axis=0.1*self.LF*self.V2
            self.MvFrm3.pos=[self.cm,self.LF*self.V3]
            self.MvTip3.pos=self.LF*self.V3; 
            self.MvTip3.axis=0.1*self.LF*self.V3
            
        tip1 = self.LF*self.V1; 
        tip1_ax = 0.1*self.LF*self.V1
        tip2 = self.LF*self.V2; 
        tip2_ax = 0.1*self.LF*self.V2
        tip3 = self.LF*self.V3; 
        tip3_ax = 0.1*self.LF*self.V3
        
        ### input trajectory to be recovered and predicted by the reservoir
        self.inputs_signals.append([list(tip1), list(tip2), list(tip3), list(tip1_ax), list(tip2_ax), list(tip3_ax)])
        self.teach_signals.append([list(tip1), list(tip1_ax)])

        # Toggle Painting of orbit and the L w vectors
        if self.window.mouse.events:
            self.mouseObj = self.window.mouse.getevent()
            self.window.mouse.events = 0
            if self.mouseObj.click == "left":   # Toggle orbit of x1
                if self.PaintOrbit == True:
                    self.PaintOrbit = False
                    self.tip.pos=[]
                else: self.PaintOrbit = True
            if self.mouseObj.click == "right":  # Togle display of L and w
                self.ShowVectors = 1 - self.ShowVectors
                self.L_body.visible = self.ShowVectors
                self.L_tip.visible = self.ShowVectors
                self.L_label.visible = self.ShowVectors
                self.w_body.visible = self.ShowVectors
                self.w_tip.visible = self.ShowVectors
                self.w_label.visible = self.ShowVectors 

        # Paint orbit of x1 tip and the angular velocity w        
        if self.PaintOrbit == True:
            self.tip.append(pos=self.MvTip1.pos+self.MvTip1.axis)

        w = self.omega0*(cos(self.Omega_e*self.t)*self.V1+sin(self.Omega_e*self.t)*self.V2)+self.omega3*self.V3
        w = w/mag(w)*self.LF
        self.w_body.pos = [self.cm,w]
        self.w_tip.pos=w; self.w_tip.axis=0.1*w
        self.w_label.pos= self.Char_omega(self.w_tip.pos+1.2*self.w_tip.axis)
        self.this_t += 1    

    def go_sync_pix(self):
        for y in range(3):
            this_y = (2.5 - 2.0) * np.random.rand() + 2.0
            for x in range(3):
                this_x = (3 - 2.6) * np.random.rand() + 2.6
                for z in np.linspace(-3,3,20):
                    this_z = (3 + 3) * np.random.rand() - 3
                    bw = np.random.choice([0,1])
                    randomcolor = (bw, bw, bw)
                    self.plot_pix(this_x,this_y,this_z,randomcolor)
    
    def run(self, counts, framerate=60, neu_sync = 255, omegas = None):
        '''
        counts -> timer count
        framerate -> guess
        neu_sync -> neuron that we use as a sync signal
        different_traj -> sample init omegas on a sphere
        '''
    
        if omegas != None:
            self.start_position(omega0=omegas[0],omega1=omegas[1], omega3=omegas[2])
        else:
            self.start_position() #make it start from the same initial conditions
            
        for i in range(2):
            rate(framerate)
            self.MovingFrame() 
            self.update_ellipsoid()
        tt.sleep(0.4)
        duation_in_ms = (counts/framerate)*1000+50
        
        #record spikes
        if(not self.BuildOffline):
            self.setup.mapper._program_detail_mapping(2**6) #swith on retina mapping
            t = Thread(target=self._record_chip_activity, args=(duation_in_ms,neu_sync))
            t.daemon = True
            t.start()
            outputs = []
            inputs = []
        #sync stim 
        
        ### Simple example: Give every pixel a random color:
        start = time.time()
        for i in range(self.sync_dur):
            rate(framerate)
            self.go_sync_pix()
        time_exc = tt.time() - start  
        print "##### time for sync", time_exc    
            
        start = time.time()    
        #go with rotation of ellipsoide
        while self.this_t < counts:
            self.t += self.Dt
            self.MovingFrame() # The new unit vectors
            rate(framerate)
            self.update_ellipsoid()  
        time_exc = tt.time() - start  
        print "##### time for rotation", time_exc    
           
             
        #wait record spikes thread      
        if(not self.BuildOffline):
            t.join()
            this_out = self.out[self.chip_channel].raw_data()
            index_after_start = this_out[:,0] >= self.out[self.chip_channel].t_start
            this_out_f = this_out[index_after_start,:]
            this_out_f[:,0] = this_out_f[:,0]-np.min(this_out_f[:,0])
            this_in = self.out[self.retina_channel].raw_data()
            index_after_start = this_in[:,0] >= self.out[self.retina_channel].t_start
            this_in_f = this_in[index_after_start,:]
            this_in_f[:,0] = this_in_f[:,0]-np.min(this_in_f[:,0])
            outputs.append(this_out_f)
            inputs.append(this_in_f)
            self.setup.mapper._program_detail_mapping(2**7) #switch off retina mapping

		
            return inputs, outputs
        

#main loop
if __name__ == '__main__':    

    duration = 300
    win = RetinaInputs(1,BuildOffline=True)                   
    win.run(duration)    
    inputs = np.array(win.inputs_signals)
    
    import pylab as pl
    import matplotlib
    dur, coord, space = np.shape(inputs)

    pl.figure()
    pl.title('inputs')
    for this_coord in range(coord):
        for this_dim in range(space):
            pl.plot(inputs[:,this_coord, this_dim])
    pl.show()
          
          

                
#from __future__ import division
#from visual import *
#import numpy as np

#XMAX = 400 # x and y range over 0 to XMAX
#scene.width = XMAX
#scene.fov = 0.01 # make effectively 2D
#scene.range = XMAX/2
#scene.center = (XMAX/2,XMAX/2)
#haspoints = False # assume no "points" object (Visual 4)

#scene.height = XMAX+60 # titlebar plus toolbar 60 pixels high
#pixels = points()
#haspoints = True


#def plot(nx,ny,c):
#    if haspoints:
#        pixels.append(pos=(nx,ny,0), color=c)
#    else:
#        lines[ny].color[2*nx] = c
#        lines[ny].color[2*nx+1] = c


#### Simple example: Give every pixel a random color:
#for y in range(XMAX):
#    for x in range(XMAX):
#        randomcolor = (np.random.rand(),np.random.rand(),np.random.rand())
#        plot(x,y,randomcolor)


        

