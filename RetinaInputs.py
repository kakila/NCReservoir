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

class RetinaInputs():

    def __init__(self, setup, I1 = 1.0, I3 = 2.0,
                omega0=1.0, omega1=1.0, omega3 = 1.0,
                display_arms = False,
                ShowVectors = 0,
                ShowOrbits = True,
                L1 = 1.0,
                L3 = 3.0):
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
                
        self.raw_data = []
        self._init_everything()

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
        self.window = display(title="Retina Stimulus", width=600, height=600)
        self.window.fullscreen = 0      # Change to 0 to get a floating window
        self.window.userspin = 0        # No rotation with mouse
        self.window.range = (4,4,4)
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
        label(text = "I3/I1 = %.2f    w3/w0 = %.2f" % Info, pos=(0,0,-2.5),
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
        self.Dt = 0.02/self.Omega_p
        self.PaintOrbit = False
        self.CosTheta = I3*self.omega3/sqrt((I1*self.omega0)**2+(I3*self.omega3)**2)
        self.SinTheta = sqrt(1-self.CosTheta**2)
        self.cm = vector(0,0,0)

        self.this_t = 0

    def _record_chip_activity(self, duration):
        out = self.setup.stimulate({}, send_reset_event=False, duration=duration) 
        out = out[0]
        #sync data with sync neuron
        self.raw_data = out.raw_data()
        self.raw_data[:,0] = self.raw_data[:,0]-np.min(self.raw_data[:,0])

        return

    def run(self, counts, framerate=125):
        self.start_position() #make it start from the same initial conditions
        duation_in_ms = (counts/framerate)*1000+50
        self.setup.mapper._program_detail_mapping(2**6) #swith on retina mapping
        t = Thread(target=self._record_chip_activity, args=(duation_in_ms,))
        t.daemon = True
        t.start()
        outputs = []
        while self.this_t < counts:
            rate(framerate)
            
            self.t += self.Dt
            self.MovingFrame() # The new unit vectors

            # Update the body and the moving frame
            self.Body.axis=self.L3*self.V3
            if self.display_arms:
                self.Arm1.pos=-1.2*self.L1*self.V1; self.Arm1.axis=2.4*self.L1*self.V1
                self.Arm2.pos=-1.2*self.L1*self.V2; self.Arm2.axis=2.4*self.L1*self.V2

            if self.ShowOrbits :
                self.MvFrm1.pos=[self.cm,self.LF*self.V1]
                self.MvTip1.pos=self.LF*self.V1; self.MvTip1.axis=0.1*self.LF*self.V1
                self.MvFrm2.pos=[self.cm,self.LF*self.V2]
                self.MvTip2.pos=self.LF*self.V2; self.MvTip2.axis=0.1*self.LF*self.V2
                self.MvFrm3.pos=[self.cm,self.LF*self.V3]
                self.MvTip3.pos=self.LF*self.V3; self.MvTip3.axis=0.1*self.LF*self.V3

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

        t.join()
        outputs.append(self.raw_data)
        self.setup.mapper._program_detail_mapping(2**7) #switch off retina mapping
        
        return 0, outputs

#main loop
if __name__ == '__main__':    

    win = RetinaInputs()                   
    win.run()
        
                             

        

        

        

