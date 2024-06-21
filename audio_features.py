from Feature_Extractor import extract_features

class AudioFeatures:
    def __init__(self, filename, directory):
        self.features = extract_features(filename=filename)

    def A1(self):
        return self.features["Spectral Centroid Overall Standard Deviation"]

    def A2(self):
        return self.features["Derivative of Spectral Rolloff Point Overall Standard Deviation"]

    def A3(self):
        return self.features["Derivative of Zero Crossings Overall Standard Deviation"]

    def A4(self):
        return self.features["Derivative of Strongest Frequency Via Zero Crossings Overall Standard Deviation"]
    
    def A5(self):
        return self.features["Derivative of Strongest Frequency Via Spectral Centroid Overall Standard Deviation"]

    def B1(self):
        return self.features["Standard Deviation of Zero Crossings Overall Standard Deviation"]

    def B2(self):
        return self.features["Derivative of Strongest Frequency Via Spectral Centroid Overall Standard Deviation"]

    def B3(self):
        return self.features["Derivative of Running Mean of Strongest Frequency Via Spectral Centroid Overall Standard Deviation"]

    def B4(self):
        return self.features["Strongest Frequency Via FFT Maximum Overall Standard Deviation"]

    def B5(self):
        return self.features["Derivative of Strongest Frequency Via FFT Maximum Overall Standard Deviation"]

    def feature_Entropy(self):
        return None
