# ----------------------------------------------- Import environment ----------------------------------------------------#
import numpy as np

###########################################################################################################################
# Define a class to receive the characteristics of each line detection                                                    #
###########################################################################################################################
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
    
    # Build the look ahead filter 
    def add_fit(self, fit, ploty, curve_rad):
        # Set up hyperparameter
        curvrad_diff_threshold = 150
        line_base_diff_threshold = 50

        if not self.detected:
            if len(fit) != 0:
                # if previous is not detected, we direct update line's info with data by hand 
                self.current_fit = fit
                self.radius_of_curvature = curve_rad
                self.detected = True
                self.line_base_pos = fit[0]*ploty[-1]**2 + fit[1]*ploty[-1] + fit[2]
        else:
            # Calculate the difference between the current line base and the previous line base
            curr_line_base = fit[0]*ploty[-1]**2 + fit[1]*ploty[-1] + fit[2]
            line_base_diff = np.absolute(self.line_base_pos - curr_line_base)

            # Calculate the difference between the current curvature and the previous curvature
            curvrad_diff = np.absolute(curve_rad - self.radius_of_curvature)

            # Check that they have similar curvature and line base point, we only update the line info when their difference 
            # is within some certain range
            if (curvrad_diff < curvrad_diff_threshold) & (line_base_diff < line_base_diff_threshold):
                self.current_fit = fit
                self.radius_of_curvature = curve_rad
            