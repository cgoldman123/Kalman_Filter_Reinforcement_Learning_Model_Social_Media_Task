This is code from the following article:
Fast solutions for the first-passage distribution of diffusion models with space-time-dependent drift functions and time-dependent boundaries (Boehm et al, 2021)
https://www.sciencedirect.com/science/article/pii/S0022249621000833
While it gives us the cumulative density function for a go response for different time points, it doesn't give us the PDF, so we would have to use some sort of binning method to fit (e.g., chi-sq method
from "Estimating parameters of the diffusion model:Approaches to dealing with contaminant reaction times and parameter variability" (Ratcliff, 2002).