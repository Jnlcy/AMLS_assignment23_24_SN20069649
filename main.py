from A.train_and_eval_A import *


def main():

    '''Task A: The3re are two options: 
    (1) Select,train and save a new model according to data and then evaluate it
    (2) Predict and evaluate using the already trained model'''
    
    
    print("Welcome to task A: Pneumonia classification")

    while True:
        print("\nOptions:")
        print("1. Train a new model")
        print("2. Deploy a pre-trained model")
        print("3. Exit")

        choice = input("Please enter your choice (1/2/3): ")

        if choice == '1':
            print("\nTraining a new model...")
            train_and_save_classify()
        elif choice == '2':
            print("\nDeploying a pre-trained model...")
            load_trained_model()
        elif choice == '3':
            print("\nExiting the Pneumonia Detection System.")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")
    

if __name__ == "__main__":
    main()