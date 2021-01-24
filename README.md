# DogClassificationAPI
Dog classification REST API. Receive images and determines probabilities about 5 breeds with percentage chance.

This API uses pretrained model, so this is fast instruction how to get started:<br/>
1. Download dataset from [link to dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)
2. Train your model using running: `./train_and_test_model.py`
3. Run your API: `./api.py` and use it.
4. You can use my [public android app](https://github.com/patrykce/dog-breed-classification-mobile-app), that was created for sending photos to remote API and receive results.


Result from this API is contains 5 best matching dogs for incoming pictures with percentage chance for each result</br>
</br>
Dataset info: </br>
After downloading dataset you can notice that there are folders:
- `Dog Breed Identification/tests`
- `Dog Breed Identification/train`


I used only second one(for training and testing) because only this was labeled.

After training, you can see charts, and if you want see logs in tensorboard type: `tensorboard --logdir==logs
`