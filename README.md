# Applying machine learning to investigate long-term insect–plant interactions preserved on digitized herbarium specimens
Pytorch implementation of our method for adapting deep learning method to detect different kinds of leaf damage. Our detection method is based on SSD: Single Shot MultiBox Detecto.

## Paper
[Applying Machine Learning to Investigate Long Term Insect-Plant Interactions Preserved on Digitized Herbarium Specimens](https://www.biorxiv.org/content/10.1101/790899v1)
Please cite our paper if you find it useful.
```
@article {Meineke790899,
	author = {Meineke, E.K. and Tomasi, C. and Yuan, S. and Pryer, K.M.},
	title = {Applying Machine Learning to Investigate Long Term Insect-Plant Interactions Preserved on Digitized Herbarium Specimens},
	elocation-id = {790899},
	year = {2019},
	doi = {10.1101/790899},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Premise of the study Despite the economic importance of insect damage to plants, long-term data documenting changes in insect damage ({\textquoteleft}herbivory{\textquoteright}) and diversity are limited. Millions of pressed plant specimens are now available online for collecting big data on plant-insect interactions during the Anthropocene.Methods We initiated development of machine learning methods to automate extraction of herbivory data from herbarium specimens. We trained an insect damage detector and a damage type classifier on two distantly related plant species. We experimented with 1) classifying six types of herbivory and two control categories of undamaged leaf, and 2) detecting two of these damage categories for which several hundred annotations were available.Results Classification models identified the correct type of herbivory 81.5\% of the time. The damage classifier was accurate for categories with at least one hundred test samples. We show anecdotally that the detector works well when asked to detect two types of damage.Discussion The classifier and detector together are a promising first step for the automation of herbivory data collection. We describe ongoing efforts to increase the accuracy of these models to allow other researchers to extract similar data and apply them to address a variety of biological hypotheses.},
	URL = {https://www.biorxiv.org/content/early/2019/10/02/790899},
	eprint = {https://www.biorxiv.org/content/early/2019/10/02/790899.full.pdf},
	journal = {bioRxiv}
}

```
