from crack_feature import get_crack_percentage
from depreciation_stuff import depreciation


# Load all the necessary packages
# Call the preprocess file and get the cracked image

# Call the binary file which converts the cropped image to binary file

# Call the crack_feature file to get the crack percentage
percentage_crack = get_crack_percentage("concrete_crack_images/00000019.jpg")
print(percentage_crack)
# Call the estimator code once
estimated_price = depreciation("iphone 7",3,percentage_crack/100)

# Output the Value
print(estimated_price)

# Optional add gui interface


