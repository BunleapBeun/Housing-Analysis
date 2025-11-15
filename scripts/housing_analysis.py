import pandas as pd
import numpy as np
import os

def create_housing_analysis():
    """Create comprehensive housing analysis - console output only"""
    
    print("Starting housing analysis...")
    print("=" * 50)
    
    # Multiple possible paths to find the CSV file
    possible_paths = [
        os.path.join('data', 'Housing.csv'),  # Relative path from project root
        os.path.join('..', 'data', 'Housing.csv'),  # From scripts folder
        'Housing.csv',  # Direct in current directory
    ]

    df = None
    
    # Try each possible path
    for data_path in possible_paths:
        try:
            print(f"Trying to load data from: {data_path}")
            df = pd.read_csv(data_path)
            print(f"Successfully loaded data from: {data_path}")
            break
        except FileNotFoundError:
            continue

    # If still not found, show error
    if df is None:
        print("ERROR: Could not find Housing.csv file!")
        print("Please make sure the file exists in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nCurrent working directory:", os.getcwd())
        return
    
    print(f"Dataset loaded with {len(df)} records")
    print("=" * 50)
    
    # Display basic dataset info
    display_basic_info(df)
    
    # Display distributions
    display_distributions(df)
    
    # Display price analysis
    display_price_analysis(df)
    
    # Display area analysis
    display_area_analysis(df)
    
    # Display categorical features analysis
    display_categorical_analysis(df)
    
    print("=" * 50)
    print("Analysis completed successfully!")

def display_basic_info(df):
    """Display basic information about the dataset"""
    print("\nBASIC DATASET INFORMATION")
    print("-" * 30)
    print(f"Total number of houses: {len(df):,}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    
    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head().to_string())

def display_distributions(df):
    """Display distributions of housing characteristics"""
    print("\nHOUSING CHARACTERISTICS DISTRIBUTION")
    print("-" * 40)
    
    total = len(df)
    
    # Bedrooms distribution
    print("\n1. BEDROOMS DISTRIBUTION:")
    bedroom_counts = df['bedrooms'].value_counts().sort_index()
    for bedrooms, count in bedroom_counts.items():
        percentage = (count / total) * 100
        print(f"   {bedrooms} bedrooms: {count:3d} houses ({percentage:5.1f}%)")
    
    # Bathrooms distribution
    print("\n2. BATHROOMS DISTRIBUTION:")
    bathroom_counts = df['bathrooms'].value_counts().sort_index()
    for bathrooms, count in bathroom_counts.items():
        percentage = (count / total) * 100
        print(f"   {bathrooms} bathrooms: {count:3d} houses ({percentage:5.1f}%)")
    
    # Stories distribution
    print("\n3. STORIES DISTRIBUTION:")
    stories_counts = df['stories'].value_counts().sort_index()
    for stories, count in stories_counts.items():
        percentage = (count / total) * 100
        print(f"   {stories} stories: {count:3d} houses ({percentage:5.1f}%)")
    
    # Parking distribution
    print("\n4. PARKING SPACES DISTRIBUTION:")
    parking_counts = df['parking'].value_counts().sort_index()
    for parking, count in parking_counts.items():
        percentage = (count / total) * 100
        print(f"   {parking} parking spaces: {count:3d} houses ({percentage:5.1f}%)")

def display_price_analysis(df):
    """Display price distribution analysis"""
    print("\nPRICE ANALYSIS")
    print("-" * 20)
    
    total = len(df)
    
    # Basic price statistics
    print(f"Minimum price: ₹{df['price'].min():,}")
    print(f"Maximum price: ₹{df['price'].max():,}")
    print(f"Average price: ₹{df['price'].mean():,.0f}")
    print(f"Median price: ₹{df['price'].median():,.0f}")
    
    # Price ranges
    print("\nPRICE RANGES DISTRIBUTION:")
    price_ranges = [0, 3000000, 6000000, 9000000, 12000000, 15000000]
    price_labels = ['<3M', '3M-6M', '6M-9M', '9M-12M', '12M+']
    
    df['price_range'] = pd.cut(df['price'], bins=price_ranges, labels=price_labels)
    price_dist = df['price_range'].value_counts().sort_index()
    
    for price_range, count in price_dist.items():
        percentage = (count / total) * 100
        print(f"   {price_range}: {count:3d} houses ({percentage:5.1f}%)")
    
    # Top 5 most expensive houses
    print("\nTOP 5 MOST EXPENSIVE HOUSES:")
    top_expensive = df.nlargest(5, 'price')[['price', 'area', 'bedrooms', 'bathrooms']]
    for idx, row in top_expensive.iterrows():
        print(f"   ₹{row['price']:,} - {row['area']} sqft, {row['bedrooms']} bed, {row['bathrooms']} bath")

def display_area_analysis(df):
    """Display area distribution analysis"""
    print("\nAREA ANALYSIS")
    print("-" * 15)
    
    total = len(df)
    
    # Basic area statistics
    print(f"Minimum area: {df['area'].min():,} sqft")
    print(f"Maximum area: {df['area'].max():,} sqft")
    print(f"Average area: {df['area'].mean():,.0f} sqft")
    print(f"Median area: {df['area'].median():,.0f} sqft")
    
    # Area ranges
    print("\nAREA RANGES DISTRIBUTION:")
    area_ranges = [0, 3000, 6000, 9000, 12000, 15000, 18000]
    area_labels = ['<3K', '3K-6K', '6K-9K', '9K-12K', '12K-15K', '15K+']
    
    df['area_range'] = pd.cut(df['area'], bins=area_ranges, labels=area_labels)
    area_dist = df['area_range'].value_counts().sort_index()
    
    for area_range, count in area_dist.items():
        percentage = (count / total) * 100
        print(f"   {area_range}: {count:3d} houses ({percentage:5.1f}%)")
    
    # Top 5 largest houses
    print("\nTOP 5 LARGEST HOUSES:")
    top_largest = df.nlargest(5, 'area')[['area', 'price', 'bedrooms', 'bathrooms']]
    for idx, row in top_largest.iterrows():
        print(f"   {row['area']:,} sqft - ₹{row['price']:,}, {row['bedrooms']} bed, {row['bathrooms']} bath")

def display_categorical_analysis(df):
    """Display analysis of categorical features"""
    print("\nCATEGORICAL FEATURES ANALYSIS")
    print("-" * 35)
    
    total = len(df)
    
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                         'airconditioning', 'prefarea', 'furnishingstatus']
    
    for col in categorical_columns:
        print(f"\n{col.upper()}:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"   {value}: {count:3d} houses ({percentage:5.1f}%)")

if __name__ == "__main__":
    create_housing_analysis()