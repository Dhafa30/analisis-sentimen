import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# IndoBERT values
# Top-Left (TP) = 93, Top-Right (FN) = 7
# Bottom-Left (FP) = 5, Bottom-Right (TN) = 95
cm_indobert = np.array([[93, 7],
                        [5, 95]])

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_indobert, display_labels=['Positif (1)', 'Negatif (0)'])
disp1.plot(cmap='Greens', values_format='d')
plt.title('Confusion Matrix - Evaluasi IndoBERT')
plt.savefig('confusion_matrix_indobert.png', dpi=300, bbox_inches='tight')
plt.close()

# VADER values
# Top-Left (TP) = 83, Top-Right (FN) = 23
# Bottom-Left (FP) = 25, Bottom-Right (TN) = 69
cm_vader = np.array([[83, 23],
                     [25, 69]])

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_vader, display_labels=['Positif (1)', 'Negatif (0)'])
disp2.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - VADER')
plt.savefig('confusion_matrix_vader.png', dpi=300, bbox_inches='tight')
plt.close()

print("Images regenerated successfully!")
