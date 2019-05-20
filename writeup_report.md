# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[hist_all]: ./report_imgs/all_hist.png "Histogram of all recordings"
[hist_straight]: ./report_imgs/rec2_hist.png "Histogram of normal driving recording"
[hist_correction]: ./report_imgs/rec8_hist.png "Histogram of correction recording"
[center_driving]: ./report_imgs/center_driving.jpg "Center driving"
[recover_1]: ./report_imgs/recover_1.jpg "Recovery Image"
[recover_2]: ./report_imgs/recover_2.jpg "Recovery Image"
[recover_3]: ./report_imgs/recover_3.jpg "Recovery Image"

## Rubric Points
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode. The command line option `--speed <s>` can be used to set the vehicle target speed.
* `util.py` contains functions for reading the data recordings and plotting histograms of the recorded data.
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

This model works well up to 30 mph (it is slightly oscillatory at this speed though). To test the model at this speed, use:
```sh
python drive.py model.h5 --speed 30
```


#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of two shallow convolution layers (5x5 with 6 filters) and one fully connected layer. Relu and maxpooling layers are used to introduce non-linearities. The whole architecture can be seen between lines 110 and 120 of `model.py`.

There are two preprocessing steps: the image is normalized and clipped. The clipping is done to remove the part above the horizon and the bonnet. Only the center camera images are used (the results were good enough without having to use the other cameras). These operations are on lines 111 and 112.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, a large validation set was used (30% of the overall data) and dropout layers were included between the fully connected layers (line 119).

Also, the training set was extended by using two "tricks":
* Every front camera image on the training set is flipped and added to the training set (the steering angle is flipped as well)
* The two lateral cameras are used to generate artificial steering data. They are added to the training set as center images with 0.15 of additional steering (to the left if it is a right camera image and to the right otherwise).

Finally, as the recordings are skewed for center lane driving, all points between -0.25 and 0.25 of steering were decimated by 5. The histogram below shows the distribution of steering angles for all the recordings.

![alt text][hist_all]

Finally, the model was tested by running it through the simulator and ensuring that the vehicle could stay on track. The movie of the test is on file `video.mp4`.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 62). Early stopping and checkpointing of the best model were used to increase the learning speed and effectiveness (lines 123 to 128).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I started with about 5 laps of center lane driving, plus additional data of the vehicle recovering from the left and right. After a few tests, I found out that more training data was needed in some hard parts of the track (especially entering and leaving the bridge and avoiding the dirty shoulder areas), so I recorded specific cases for those areas.

Below two histograms can be seen. The first one is of typical center lane driving. The second one show a recording focusing on recovers.

![alt text][hist_straight]

![alt text][hist_correction]



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple model and expand it slowly. I started from the first suggestion on the project description, which already gave quite impressive results, even without using non-linearities on the fully connected layers (which I found quite surprising).

I noticed that the performance of the network was far more dependent on the quality of the training data, so I spent some time "playing" the game and recording only high quality laps.

I expected the problem to be relatively simple, as it is a simple proportional controller with a little bit of feed-forward. I expect that, by knowing how well centered the vehicle is on the road, how its yaw is aligned to the road, and how well it will be centered in the next 15 meters, a simple controller could be made that creates very good results. Therefore, I expected a relatively low complexity network could handle the task quite well.

After a few tries, I found out that the model was quite prone to overfitting and that the performance of the model varied a lot depending from one training run to the other. To fix it, I added the "artificial case generation" from the left and right cameras and the removal of center lane driving cases to compensate for the fact that they are overrepresented. Also, I added 50% dropout on the fully connected layers.

This generated a strong improvement on the performance of the network, so much that I started playing with the removal of layers. After some experiments, I found that I could remove one of the fully connected layers without affecting testing performance.


The final model is able to drive the course well up to a speed of 30. Above 20 mph, it becomes slightly oscillatory but still performs well.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 110-120) consists of a CNN with the following layers:

| Layer (type) | Output shape  | # Param | Details                    |
| :---------   | :----------:  | :------ | :--------------            |
| Lambda       | (160, 320, 3) |       0 | Normalize image            |
| Cropping 2D  | (65, 32, 3)   |       0 |                            |
| Conv2D       | (61, 316, 6)  |     456 | 5x5 filters                |
| MaxPooling2D | (30, 158, 6)  |       0 | 2x2 pooling and stride     |
| Conv2D       | (26, 154, 6)  |     906 | 5x5 filters                |
| MaxPooling2D | (13, 77, 6)   |       0 | 2x2 pooling and stride     |
| Flatten      | (6006,)       |       0 |                            |
| Dense        | (120, )       |  720840 |                            |
| Dropout      | (120,)        |       0 | Set to 50% during training |
| Dense        | (1,)          |     121 |                            |

For a total of 722,323 trainable parameters.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded about five laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and also to recognize uncommon lane boundaries (like the dirty ones and the ones near the bridge). These images show what a recovery looks like starting from the sharp corners:

![alt text][recover_1]
![alt text][recover_2]
![alt text][recover_3]

To augment the data sat, I flipped images and steering, to make sure that the network would not be biased for left turns. I also used the left and right cameras to generate artificial additional recovery data.

All augmentation is done on the training set, inside the generator. Due to the large number of images (about 5000 on the training set after decimation), a generator is used. The generator is in file `model.py`, from line 11 to 63.

The data set is split in about 70% of the original decimated data for training and the remaining 30% for testing. The split is done after decimation, to avoid biasing the validation set. Also, the split is random, so every run will have a slightly different training and validation set sizes.

I used an adam optimizer so that manually defining the learning rate wasn't necessary. Also, I used the `EarlyStopping` and `ModelCheckpoint` callbacks to automatically stop training after it converged and to get the best trained model. The training typically stopped around 30 epochs.

As can be seen below, training time and memory usage are reasonable.

```sh
/usr/bin/time -v python model.py

Total number of images: 20975
Total number of original training : 5177
Number of validation samples: 1672

[....]

162/162 [==============================] - 8s 49ms/step - loss: 0.0941 - val_loss: 0.0546
Epoch 2/50
162/162 [==============================] - 7s 43ms/step - loss: 0.0682 - val_loss: 0.0479
Epoch 3/50
162/162 [==============================] - 6s 34ms/step - loss: 0.0582 - val_loss: 0.0427
Epoch 4/50
162/162 [==============================] - 5s 33ms/step - loss: 0.0552 - val_loss: 0.0432
Epoch 5/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0545 - val_loss: 0.0417
Epoch 6/50
162/162 [==============================] - 5s 33ms/step - loss: 0.0527 - val_loss: 0.0404
Epoch 7/50
162/162 [==============================] - 5s 33ms/step - loss: 0.0490 - val_loss: 0.0369
Epoch 8/50
162/162 [==============================] - 6s 34ms/step - loss: 0.0502 - val_loss: 0.0365
Epoch 9/50
162/162 [==============================] - 7s 42ms/step - loss: 0.0456 - val_loss: 0.0372
Epoch 10/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0445 - val_loss: 0.0383
Epoch 11/50
162/162 [==============================] - 6s 35ms/step - loss: 0.0486 - val_loss: 0.0346
Epoch 12/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0429 - val_loss: 0.0348
Epoch 13/50
162/162 [==============================] - 6s 34ms/step - loss: 0.0408 - val_loss: 0.0343
Epoch 14/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0423 - val_loss: 0.0359
Epoch 15/50
162/162 [==============================] - 6s 35ms/step - loss: 0.0386 - val_loss: 0.0406
Epoch 16/50
162/162 [==============================] - 7s 42ms/step - loss: 0.0453 - val_loss: 0.0336
Epoch 17/50
162/162 [==============================] - 7s 43ms/step - loss: 0.0390 - val_loss: 0.0337
Epoch 18/50
162/162 [==============================] - 6s 35ms/step - loss: 0.0353 - val_loss: 0.0363
Epoch 19/50
162/162 [==============================] - 6s 36ms/step - loss: 0.0402 - val_loss: 0.0330
Epoch 20/50
162/162 [==============================] - 6s 34ms/step - loss: 0.0377 - val_loss: 0.0327
Epoch 21/50
162/162 [==============================] - 7s 42ms/step - loss: 0.0357 - val_loss: 0.0337
Epoch 22/50
162/162 [==============================] - 6s 35ms/step - loss: 0.0356 - val_loss: 0.0315
Epoch 23/50
162/162 [==============================] - 7s 43ms/step - loss: 0.0373 - val_loss: 0.0347
Epoch 24/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0354 - val_loss: 0.0305
Epoch 25/50
162/162 [==============================] - 7s 42ms/step - loss: 0.0321 - val_loss: 0.0306
Epoch 26/50
162/162 [==============================] - 6s 34ms/step - loss: 0.0323 - val_loss: 0.0288
Epoch 27/50
162/162 [==============================] - 6s 35ms/step - loss: 0.0318 - val_loss: 0.0315
Epoch 28/50
162/162 [==============================] - 7s 41ms/step - loss: 0.0351 - val_loss: 0.0311
Epoch 29/50
162/162 [==============================] - 7s 43ms/step - loss: 0.0291 - val_loss: 0.0315
Epoch 30/50
162/162 [==============================] - 5s 34ms/step - loss: 0.0326 - val_loss: 0.0319
Epoch 31/50
162/162 [==============================] - 6s 36ms/step - loss: 0.0314 - val_loss: 0.0295
Epoch 32/50
162/162 [==============================] - 6s 40ms/step - loss: 0.0319 - val_loss: 0.0305
Epoch 33/50
162/162 [==============================] - 6s 38ms/step - loss: 0.0275 - val_loss: 0.0314
Epoch 34/50
162/162 [==============================] - 6s 38ms/step - loss: 0.0323 - val_loss: 0.0308
	Command being timed: "python model.py"
	User time (seconds): 303.77
	System time (seconds): 21.86
	Percent of CPU this job got: 155%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:29.85
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 2320956
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 678012
	Voluntary context switches: 1403470
	Involuntary context switches: 67591
	Swaps: 0
	File system inputs: 7120
	File system outputs: 255120
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```
