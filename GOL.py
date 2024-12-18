import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
# Fix matplotlib import by installing/upgrading matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install/upgrade matplotlib using: pip install --upgrade matplotlib")
    raise
from sklearn.metrics import mean_squared_error, r2_score

class GameOfLifePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=2)
        
    def create_random_board(self, size):
        return np.random.choice([0, 1], size=(size, size))
    
    def count_alive_cells(self, board):
        return np.sum(board)
    
    def get_neighbors(self, board, x, y):
        rows, cols = board.shape
        total = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                row = (x + i) % rows
                col = (y + j) % cols
                total += board[row][col]
        return total
    
    def next_generation(self, board):
        rows, cols = board.shape
        new_board = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                neighbors = self.get_neighbors(board, i, j)
                if board[i][j] == 1:
                    if neighbors in [2, 3]:
                        new_board[i][j] = 1
                else:
                    if neighbors == 3:
                        new_board[i][j] = 1
        return new_board
    
    def simulate_until_death(self, board):
        generations = 0
        current_board = board.copy()
        previous_states = []
        
        while True:
            generations += 1
            current_board = self.next_generation(current_board)
            alive_cells = self.count_alive_cells(current_board)
            
            # Check if all cells are dead
            if alive_cells == 0:
                return generations
            
            # Check for repeating patterns
            board_hash = current_board.tobytes()
            if board_hash in previous_states:
                return generations
            
            previous_states.append(board_hash)
            
            # Prevent infinite loops
            if generations > 1000:
                return 1000
    
    def extract_features(self, board):
        features = [
            self.count_alive_cells(board),  # Total alive cells
            np.sum(board[::2, ::2]),        # Checkerboard pattern
            np.sum(board[1::2, 1::2]),      # Alternate checkerboard
            np.sum(board[0]),               # First row
            np.sum(board[:, 0]),            # First column
            np.sum(np.diag(board))          # Diagonal
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, num_samples=1000, board_size=10):
        X = []
        y = []
        
        for _ in range(num_samples):
            board = self.create_random_board(board_size)
            features = self.extract_features(board)
            generations = self.simulate_until_death(board)
            
            X.append(features[0])
            y.append(generations)
        
        X = np.array(X)
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
    
    def predict_generations(self, board):
        features = self.extract_features(board)
        features_poly = self.poly.transform(features)
        prediction = self.model.predict(features_poly)[0]
        return max(1, int(round(prediction)))
    
    def plot_predictions(self, y_true, y_pred, title):
        plt.figure(figsize=(10, 6))
        # Convert single values to arrays if needed
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('True Generations')
        plt.ylabel('Predicted Generations')
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    predictor = GameOfLifePredictor()
    print("Training the model...")
    predictor.train(num_samples=1000, board_size=20)
    
    # Test the predictor
    test_board = predictor.create_random_board(20)
    predicted = predictor.predict_generations(test_board)
    actual = predictor.simulate_until_death(test_board)
    
    print(f"Predicted generations until death: {predicted}")
    print(f"Actual generations until death: {actual}")
    print(f"Prediction error: {abs(predicted - actual)} generations")

    # Create a custom pattern
    custom_pattern = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])

    predicted_generations = predictor.predict_generations(custom_pattern)
    print(f"This pattern will run for approximately {predicted_generations} generations")

    # Generate predictions for plotting
    X_test = predictor.extract_features(test_board)
    X_test_poly = predictor.poly.transform(X_test)
    test_pred = predictor.model.predict(X_test_poly)
    
    # Plot predictions
    predictor.plot_predictions(actual, test_pred, "Game of Life Predictions vs Actual")
