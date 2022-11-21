# ML models developed for an IPS

Improved indoor location mechanisms with ML techniques.

This github repository contains the datasets used in some experiments made with an RF antenna and multiple passive RFID tags. The main goal was the exploitation of new features that could correlate with the distance between the tags and the antenna.

The antenna controller uses the Impinj R2000 chipset supporting the EPC C1 GEN2 protocol, ISO18000-6C. This setup should be one of the most commonly used, as the hardware and chipset are very popular. This as a major contribution from this work, as the output can be ap-plied to a wide set of existing or future, deployments.

The initial test showed that RSSI was unreliable as a feature for our specific antenna. This happens because the RF an-tenna, by default, performs an automatic compensa-tion of the emission power gains, which translates into constant average values, unrelated to the known dis-tance values for the passive RFID tags arranged on site.

It is invetigated the usage of alternative features to overcome this issue and achieve reasonable accuracy without modifying the hardware. The final model uses the number of activation and the average time between activations from some power levels as the features for indoor location.

The accuracy of the developed models is on par with other solutions without using RSSI data. It obtained an error of 0.00 m within a range of 5 m in the second experiment and an error of 0.55 m within a range of 10 m in the third experiment, resulting in an average error of 0.275 m.