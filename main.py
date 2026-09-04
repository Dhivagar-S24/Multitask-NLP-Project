# ============================================
# MULTITASK NLP SYSTEM
# ============================================

import subprocess
import sys
import os


def print_header():
    print("\n")
    print("=" * 50)
    print("             MULTITASK NLP SYSTEM")
    print("=" * 50)
    print("1. Next Word Prediction")
    print("2. Perplexity Evaluation")
    print("3. Language Translation")
    print("4. Sentiment Analysis")
    print("5. Exit")
    print("=" * 50)


def run_program(filename):
    print("\n")
    print("=" * 50)
    print(f"Starting {filename}")
    print("=" * 50)
    print()

    try:
        subprocess.run([sys.executable, filename], check=True)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"\nError while running {filename}: {e}")


while True:

    print_header()

    choice = input("Enter your choice: ")

    if choice == "1":
        run_program("next_word.py")

    elif choice == "2":
        run_program("perplexity.py")

    elif choice == "3":
        run_program("translation.py")

    elif choice == "4":
        run_program("sentiment.py")

    elif choice == "5":
        print("\nThank you for using Multitask NLP System!")
        print("Exiting...")
        break

    else:
        print("\nInvalid choice!")
        print("Please enter a number from 1 to 5.")