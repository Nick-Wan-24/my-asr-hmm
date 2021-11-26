class Settings:

    def __init__(self):
        # for GMM-HMM model
        self.number_of_states = 4
        self.number_of_gaussian = 5
        self.number_of_iteration = 5

        # for MFCC computation
        self.dimension_of_vector = 39
        self.frameSize = 200
        self.overlapSize = 100
        self.N_mel_dct = 13
        self.N_mel = 26
        
        # audio file
        self.training_file_directory = '.\\Audio\\train'
        self.testing_file_directory = '.\\Audio\\test'



