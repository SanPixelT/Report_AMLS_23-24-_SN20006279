import Task_1_github
import Task_2_github

def main():
    choice = input("Enter 'A' to run Task A or 'B' to run Task B: ").strip().upper()
    
    if choice == 'A':
        print("Running Task A...")
        Task_1_github.run_task1()
    elif choice == 'B':
        print("Running Task B...")
        Task_2_github.run_task2()
    else:
        print("Invalid input. Please enter 'A' or 'B'.")

if __name__ == "__main__":
    main()