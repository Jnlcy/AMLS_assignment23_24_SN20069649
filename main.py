from A.train_and_eval_A import *
from B.load_and_eval_B import *
from B.train_and_eval_B import *


def main():


    while True:

        print('Welcome to AMLS_assignment23_24, which task which you see?')
        print('Options:')
        print('A: Binary classification')
        print('B: Multiclass classification')
        print('E. Exit')

        task = input("Please enter the task (A/B/E): ")

        if task =='A':



            '''Task A: There are two options: 
            (1) Select,train and save a new model according to data and then evaluate it
            (2) Predict and evaluate using the already trained model'''
            
            
            print("\nWelcome to task A: Pneumonia binary classification")
            
            model_filename ="pneumoniamnist_model.pkl"
            data_flag = 'pneumoniamnist'
            folder = 'A/'
            
            while True:
                print("\nOptions:")
                print("1. Train a classical machine learning model")
                print("2. Deploy a pre-trained model")
                print("3. Exit")

                choice = input("Please enter your choice (1/2/3): ")

                if choice == '1':
                    print("\nTraining a new model...")
                    train_and_save_classify()
                elif choice == '2':
                    print("\nDeploying a pre-trained model...")
                    load_trained_model(folder,model_filename,data_flag)
                elif choice == '3':
                    print("\nExiting the Pneumonia Detection System.")
                    break
                else:
                    print("\nInvalid choice. Please enter 1, 2, or 3.")
        
        elif task == 'B':

            print("Welcome to task B: Pathmnist multiclass classification")

            data_flag = 'pathmnist'
            folder = 'B/'

            model_filename = "pathmnist_model.h5"
            data_flag = 'pathmnist'

            while True:
                print("\nOptions:")
                print("1. Train a CNN model")
                print("2. Deploy a pre-trained CNN model")
                print("3. Exit")

                choice = input("Please enter your choice (1/2/3): ")

                if choice == '1':
                    print("\nTraining a new model...")
                    train_and_eval_B()
                elif choice == '2':
                    print("\nDeploying a pre-trained model...")
                    load_CNN_model(data_flag)
                elif choice == '3':
                    print("\nExiting the Pathmnist classification System.")
                    break
                else:
                    print("\nInvalid choice. Please enter 1, 2, or 3.")
        
        elif task =='E':
            print("\nExiting the Assignment.")
            break
        else:
            print("\nInvalid choice. Please enter A, B, or E.")







if __name__ == "__main__":
    main()