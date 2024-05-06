### Project description

This version of the optimization algorithm utilizes ADAM optimization for one parameter at a time. It utilizes the following dictionary which holds each parameter and its corresponding optimization status. I will initialize the system with the optimization status false for all parameters. 

```python
self.optimization_status = {
	"focus": False,
	"second_dispersion": False,
	"third_dispersion": False
}       
```

In the main `optimize_count` function I upgraded the code to utilize this dictionary to update one parameter at a time. The update of the parameter values for the second and third dispersion will only occur after the previous parameter has been optimized (focus is the initial optimization parameter).

```python
    def optimize_count(self):
        self.calc_derivatives()

        self.calc_estimated_momentum() # calc estimated biased and unbaised momentum estimates 
        self.calc_squared_grad() # calc estimated biased and unbaised squared gradient estimates 

        if not self.optimization_status["focus"]:
            if np.abs((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) > 1:
                self.new_focus = self.focus_history[-1] - ((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))      
                self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
                self.new_focus = round(self.new_focus)
                self.focus_history = np.append(self.focus_history, [self.new_focus])
                mirror_values[0] = self.new_focus

            elif np.abs(((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1:
                print("Convergence achieved in focus")
        
        if self.optimization_status["focus"] == True:
            if np.abs((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) > 1:                                            
                self.new_second_dispersion = self.second_dispersion_history[-1] - ((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))
                self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
                self.new_second_dispersion = round(self.new_second_dispersion)
                self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])
                dispersion_values[0] = self.new_second_dispersion

            elif np.abs(((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1:
                print("Convergence achieved in second dispersion")

        if self.optimization_status["focus"] == True and self.optimization_status["second_dispersion"] == True:
            if np.abs((self.third_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) > 1:
                self.new_third_dispersion = self.third_dispersion_history[-1] - ((self.third_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))
                self.new_third_dispersion = np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND)
                self.new_third_dispersion = round(self.new_third_dispersion)
                self.third_dispersion_history = np.append(self.third_dispersion_history, [self.new_third_dispersion])
                dispersion_values[1] = self.new_third_dispersion

            elif np.abs(((self.third_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1:
                print("Convergence achieved in third dispersion")
                
        # if the change in all variables is less than one (we can not take smaller steps thus this is the optimization boundry)
        if (
            np.abs(((self.third_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1 and
            np.abs(((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1 and
            np.abs(((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) < 1
        ):
            print("Convergence achieved")
                    
        if self.image_groups_processed > 2:
            # if the count is not changing much this means that we are near the peak 
            if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.count_change_tolerance:
                print("Convergence achieved")
```

Now the optimization and convergence terms have been modified to:
```python
if (np.abs((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) < 1) and self.optimization_print["focus"] == True and not self.optimization_status["second_dispersion"] and not self.optimization_print["second_dispersion"]:
	self.optimization_status["focus"] = True
	print("focus convergence achieved")
	print('-------------')

if (np.abs((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) < 1) and self.optimization_print["second_dispersion"] == True and self.optimization_status["focus"] == True and not self.optimization_status["third_dispersion"] and not self.optimization_print["second_dispersion"]:
	self.optimization_status["second_dispersion"] = True
	print("second dispersion convergence achieved")
	print('-------------')

if (np.abs((self.third_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon)) < 1) and self.optimization_print["third_dispersion"] == True and self.optimization_status["focus"] == True and self.optimization_status["second_dispersion"] == True:
	self.optimization_status["third_dispersion"] = True
	print("third dispersion convergence achieved")
	print('-------------')

if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.tolerance:
	self.optimization_status["focus"] = True
	self.optimization_status["second_dispersion"] = True
	self.optimization_status["third_dispersion"] = True
	print("convergence achieved")
	print('-------------')
```

I updated the printing to indicate which parameter is being optimized. In the code `if not self.optimization_status["focus"]` is equivalent to `if self.optimization_status["focus"] == False`. Additionally, I introduced the following dictionary to indicate the status of the printing:

```python
self.optimization_print = {
	"focus": False,
	"second_dispersion": False,
	"third_dispersion": False
}
```

Now the `process_images()` function takes the following form where the console indicates which parameter is being optimized and the code above prints a message to indicate that the parameter has been optimized.

```python
def process_images(self, new_images):
	self.initialize_image_files()
	new_images = [image_path for image_path in new_images if os.path.exists(image_path)]
	new_images.sort(key=os.path.getctime)
	for image_path in new_images:
		img_mean_count = self.calc_xray_count(image_path)
		self.n_images_count_sum += np.sum(img_mean_count)
		self.run_count += 1
		if self.run_count % self.n_images == 0:
			self.mean_count_per_n_images = np.mean(img_mean_count)
			self.count_history = np.append(self.count_history, [self.mean_count_per_n_images])
			self.n_images_run_count += 1
			self.iteration_data = np.append(self.iteration_data, [self.n_images_run_count])
			if self.n_images_run_count == 1:
				print('-------------')  

				self.focus_history = np.append(self.focus_history, [self.initial_focus])      
				self.second_dispersion_history = np.append(self.second_dispersion_history, [self.initial_second_dispersion])                  
				self.third_dispersion_history = np.append(self.third_dispersion_history, [self.initial_third_dispersion])
				self.momentum_estimate_history = np.append(self.momentum_estimate_history, [self.initial_momentum_estimate])
				self.squared_gradient_history = np.append(self.squared_gradient_history, [self.initial_squared_gradient])
				self.focus_learning_rate_history = np.append(self.focus_learning_rate_history, [self.initial_focus_learning_rate])
				self.second_dispersion_learning_rate_history = np.append(self.second_dispersion_learning_rate_history, [self.initial_second_dispersion_learning_rate])
				self.third_dispersion_learning_rate_history = np.append(self.third_dispersion_learning_rate_history, [self.initial_third_dispersion_learning_rate])
				print(f"initial values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
				print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
				self.initial_optimize()
				print(f"current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")

			if self.n_images_run_count >= 2:
				self.n_images_dir_run_count += 1
				self.optimize_count()
				if not self.optimization_status["focus"] and not self.optimization_print["focus"]:
					self.optimization_print["focus"] = True
					print("optimizing focus")
				if not self.optimization_status["second_dispersion"] and self.optimization_status["focus"] == True and not self.optimization_print["second_dispersion"]:
					self.optimization_print["second_dispersion"] = True
					print("optimizing second dispersion")
				if not self.optimization_status["third_dispersion"] and self.optimization_status["focus"] == True and self.optimization_status["second_dispersion"] == True and not self.optimization_print["third_dispersion"]:
					self.optimization_print["third_dispersion"] = True
					print("optimizing third dispersion")                      
				if not self.optimization_status["focus"]:
					print(f"mean_count_per_{self.n_images}_images {self.count_history[-1]}, current value: focus {self.focus_history[-1]}")
				if not self.optimization_status["second_dispersion"] and self.optimization_status["focus"] == True:
					print(f"mean_count_per_{self.n_images}_images {self.count_history[-1]}, current value: second_dispersion {self.second_dispersion_history[-1]}")
				if not self.optimization_status["third_dispersion"] and self.optimization_status["focus"] == True and self.optimization_status["second_dispersion"] == True:
					print(f"mean_count_per_{self.n_images}_images {self.count_history[-1]}, current value: third_dispersion {self.third_dispersion_history[-1]}")
				if self.optimization_status["third_dispersion"] == True and self.optimization_status["second_dispersion"] == True and self.optimization_status["focus"]:
					print("all parameters optimized")
			print(self.optimization_status)
			self.write_values() # write values and send via FTP connection
			self.plot_reset() # update plotting lists and reset variables
			print('-------------')
```
