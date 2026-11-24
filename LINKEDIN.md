Can a strawberry leaf’s colour tell us when it’s stressed? And does it help to know where the discolouration appears?

As part of my master’s research in smart agriculture at Jeonbuk National University, I explored those questions using greenhouse time-lapse images of 56 Seolhyang strawberry plants. I used Segment Anything to isolate annotated leaves, then used CIELAB colour rules to map yellowing and browning and estimate how much of each leaf was affected.

🍓 **Detecting visible symptoms:** The colour-zone score reached a ROC AUC of **0.81** on 34 held-out plants. Mean leaf hue performed about as well. The pixel map gave me something a single hue value couldn’t show: where the score came from.

🍓 **Tracking symptoms over time:** The score rose about **0.27 percentage points per day faster** on leaves labelled symptomatic than on leaves labelled symptom-free from the same plants.

🍓 **Distinguishing tip burn from yellowing:** Adding information about where discolouration appeared on the leaf didn’t improve the classifier. When I reviewed the 20 highest-scoring leaves labelled symptom-free, **11 were backlit or showed the leaf underside**. Both can make a leaf look yellower in an RGB image.

That audit changed what I’d prioritise next for smartphone-based crop monitoring. I’d start by checking whether the camera sees the upper leaf surface in usable light, then validate the severity maps against hand-annotated lesions.

#SmartAgriculture #ComputerVision #PlantPhenotyping #StrawberryResearch

---

Suggested carousel order: leaf mask and pixel classes, ROC curve, severity distributions, then progression on test plants. Add the GitHub link once the repository is online.
