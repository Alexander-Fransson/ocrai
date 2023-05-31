
from django.core.management.base import BaseCommand 
from triumph_backend.ocrai.ocrai import OCRAI, preprocess_image, generate_batches, purify_answers
import tensorflow as tf
from keras.backend import clear_session
from keras.losses import CategoricalCrossentropy

class Command(BaseCommand):
    
    def handle(self, *args, **kwargs):
        clear_session()
        
        question_file = '/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/media/question.png'
        answer_file = '/Users/alexander.fransson/Documents/GitHub/tastetriumph.com/backend/triumph_backend/media/answer.png'

        with open(question_file, 'rb') as qf, open(answer_file, 'rb') as af:
            question_data = qf.read()
            answer_data = af.read()

        question = preprocess_image(question_data)
        answer = preprocess_image(answer_data)

        question_batch, answer_batch = generate_batches(question, answer)

        model = OCRAI()

        model.compile(
            loss=CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        model.fit(
            question_batch,
            answer_batch,
            batch_size=2,
            epochs=5,
            verbose=2,
            class_weight='balanced'
        )

        model.save('../ocrai/nn')

        clear_session()
        
        