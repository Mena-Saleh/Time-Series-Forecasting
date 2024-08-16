import pickle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression  # You can replace this with any regression model of your choice
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def rfe_select(X, y, n = 20, estimator = LinearRegression()):
    # Initialize the RFE selector
    rfe = RFE(estimator = estimator, n_features_to_select = n)

    # Fit the RFE selector to the data
    rfe.fit(X, y)

    # Get the boolean mask of selected features
    selected_features = rfe.support_

    # Get the feature ranking (1 for selected features, 0 for eliminated features)
    ranked_features = rfe.ranking_

    # Save the selected features using pickle
    with open('Pickled/selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)

    return selected_features, ranked_features

def apply_pca(x, num_pca_components=5, is_train=True):
    if is_train:
        # Standardizing the features before applying PCA
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Apply PCA and fit to training data
        pca = PCA(n_components=num_pca_components)
        x = pca.fit_transform(x)

        # Save the fitted PCA model and scaler using pickle
        with open('Pickled/PCA.pkl', 'wb') as file:
            pickle.dump(pca, file)
        with open('Pickled/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
    else:
        # Load the saved PCA model and scaler
        with open('Pickled/PCA.pkl', 'rb') as file:
            pca = pickle.load(file)
        with open('Pickled/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        # Standardize the test data and apply the trained PCA
        x = scaler.transform(x)
        x = pca.transform(x)
    
    return x

# Example usage:
# selected_features, ranked_features = select_features_with_rfe(X_train, y_train)
# X_train_selected = X_train.loc[:, selected_features]
# X_test_selected = X_test.loc[:, selected_features]
