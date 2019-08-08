import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import neural_model.gen_captions as gen
import neural_model.train as tr
import neural_model.bleu_performance as bleu

if __name__ == "__main__":
    while True:
        choose = int(input("""
        1 for image captioning
        2 to train a model
        3 for BLEU scoring for a model
        0 to exit the app
        
        Input: 
        """))
        if choose == 0:
            print("Exiting the app...")
            break
        if choose == 1:
            inp = input("Image name: ")
            print("==========")
            gen.generate(inp)
            print("==========")
        if choose == 2:
            print("Model training")
            print("==========")
            tr.trainer()
        if choose == 3:
            print("==========")
            bleu.scoring()
