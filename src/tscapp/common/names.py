import random
random.seed()

dict_pre = [
    "Happy","Sad","Bright","Dark",
    "Fast","Slow","Loud","Quiet",
    "Strong","Weak","Tall","Short",
    "Wide","Narrow","Heavy","Light",
    "Hot","Cold","Warm","Cool",
    "Clean","Dirty","Soft","Hard",
    "Smooth","Rough","Sharp","Dull",
    "Beautiful","Ugly","Rich","Poor",
    "Young","Old","New","Ancient",
    "Modern","Simple","Complex","Easy",
    "Difficult","Brave","Cowardly","Kind",
    "Cruel","Gentle","Harsh","Honest",
    "Dishonest","Loyal","Curious","Bored",
    "Excited","Calm","Nervous","Confident",
    "Shy","Friendly","Hostile","Generous",
    "Selfish","Patient","Impatient","Cheerful",
    "Gloomy","Energetic","Tired","Healthy",
    "Sick","Clever","Foolish","Wise",
    "Ignorant","Careful","Careless","Serious",
    "Playful","Ambitious","Lazy","Talented",
    "Ordinary","Unique","Common","Rare",
    "Fresh","Stale","Sweet","Bitter",
    "Sour","Spicy","Delicious","Tasty",
    "Colorful","Plain","Elegant","Fancy",
    "Messy","Tidy","Spacious","Cramped",
    "Adventurous", "Peaceful", "Mysterious", "Vibrant",
    "Graceful", "Awkward", "Fierce", "Timid",
    "Reliable", "Unpredictable", "Charming", "Grumpy",
    "Optimistic", "Pessimistic", "Humble", "Arrogant",
    "Cautious", "Reckless", "Diligent", "Sloppy",
    "Vivid", "Faded", "Massive", "Tiny",
    "Enormous", "Minuscule", "Glorious", "Shabby",
    "Radiant", "Dim", "Sturdy", "Fragile",
    "Flexible", "Rigid", "Spontaneous", "Methodical",
    "Cheeky", "Solemn", "Witty", "Bland",
    "Mighty", "Feeble", "Luminous", "Murky",
    "Jubilant", "Melancholy", "Vigorous", "Sluggish",
    "Punctual", "Tardy", "Thoughtful", "Thoughtless",
    "Courteous", "Rude", "Eager", "Reluctant",
    "Prosperous", "Destitute", "Majestic", "Humdrum",
    "Whimsical", "Practical", "Exotic", "Familiar",
    "Turbulent", "Serene", "Robust", "Delicate",
    "Astute", "Naive", "Genuine", "Fake",
    "Immense", "Petite", "Glossy", "Matte",
    "Chilly", "Sweltering", "Damp", "Arid",
    "Nimble", "Clumsy", "Resilient", "Vulnerable",
    "Meticulous", "Haphazard", "Boisterous", "Subdued",
    "Wicked", "Virtuous", "Sly", "Forthright",
    "Zealous", "Indifferent", "Quaint", "Contemporary",
    "Steadfast", "Fickle", "Lucid", "Baffling",
]
dict_in = [
    "House", "Car", "Tree", "River",
    "Mountain", "Bridge", "Table", "Chair",
    "Book", "Phone", "Bottle", "Window",
    "Door", "Garden", "Beach", "Forest",
    "Castle", "Tower", "Church", "School",
    "Hospital", "Library", "Museum", "Park",
    "Bicycle", "Airplane", "Boat", "Train",
    "Clock", "Mirror", "Lamp", "Sofa",
    "Pillow", "Blanket", "Cup", "Plate",
    "Spoon", "Fork", "Knife", "Basket",
    "Umbrella", "Backpack", "Wallet", "Camera",
    "Computer", "Television", "Radio", "Guitar",
    "Piano", "Violin", "Drum", "Flute",
    "Street", "Alley", "Highway", "Tunnel",
    "Island", "Valley", "Desert", "Ocean",
    "Lake", "Waterfall", "Cliff", "Cave",
    "Field", "Farm", "Barn", "Fence",
    "Statue", "Fountain", "Monument", "Cathedral",
    "Stadium", "Theater", "Cinema", "Market",
    "Bakery", "Restaurant", "Cafe", "Hotel",
    "Airport", "Station", "Harbor", "Pier",
    "Factory", "Warehouse", "Office", "Bank",
    "Palace", "Cottage", "Cabin", "Apartment",
    "Shelf", "Drawer", "Closet", "Curtain",
    "Candle", "Vase", "Painting", "Sculpture",
    "Ladder", "Toolbox", "Hammer", "Wrench",
]
dict_post = [
    "Happiness", "Sadness", "Brightness", "Darkness",
    "Speed", "Slowness", "Loudness", "Quietness",
    "Strength", "Weakness", "Tallness", "Shortness",
    "Width", "Narrowness", "Heaviness", "Lightness",
    "Heat", "Coldness", "Warmth", "Coolness",
    "Cleanliness", "Dirtiness", "Softness", "Hardness",
    "Smoothness", "Roughness", "Sharpness", "Dullness",
    "Beauty", "Ugliness", "Richness", "Poverty",
    "Youth", "Oldness", "Newness", "Antiquity",
    "Modernity", "Simplicity", "Complexity", "Ease",
    "Difficulty", "Bravery", "Cowardice", "Kindness",
    "Cruelty", "Gentleness", "Harshness", "Honesty",
    "Dishonesty", "Loyalty", "Curiosity", "Boredom",
    "Excitement", "Calmness", "Nervousness", "Confidence",
    "Shyness", "Friendliness", "Hostility", "Generosity",
    "Selfishness", "Patience", "Impatience", "Cheerfulness",
    "Gloominess", "Energy", "Tiredness", "Health",
    "Sickness", "Cleverness", "Foolishness", "Wisdom",
    "Ignorance", "Carefulness", "Carelessness", "Seriousness",
    "Playfulness", "Ambition", "Laziness", "Talent",
    "Ordinariness", "Uniqueness", "Commonness", "Rarity",
    "Freshness", "Staleness", "Sweetness", "Bitterness",
    "Sourness", "Spiciness", "Deliciousness", "Tastiness",
    "Colorfulness", "Plainness", "Elegance", "Fanciness",
    "Messiness", "Tidiness", "Spaciousness", "Crampedness",
    "Boldness", "Weariness", "Clarity", "Obscurity",
]

def random_adjective(count: int = 1) -> list[str] | str:
    sample = random.sample(dict_pre, count)
    if count == 1:
        return sample[0]
    return sample

def random_in(count: int = 1) -> list[str] | str:
    sample = random.sample(dict_in, count)
    if count == 1:
        return sample[0]
    return sample

def random_post(count: int = 1) -> list[str] | str:
    sample = random.sample(dict_post, count)
    if count == 1:
        return sample[0]
    return sample

def random_noun(count: int = 1) -> list[str] | str:
    sample = random.sample(dict_in + dict_post, count)
    if count == 1:
        return sample[0]
    return sample

def random_sentence(long: bool = False) -> str:
    if long:
        return f"{random_adjective()} {random_in()} of {random_post()}"
    else:
        return f"{random_adjective()} {random_noun()}"