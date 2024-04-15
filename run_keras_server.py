import os
import flask
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import random

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Initialize Flask application
app = flask.Flask(__name__)

# Set CPU as available physical device
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the pre-trained Keras model
model_path = 'VAE_Room_Generator_Decoder_1352.h5'
model = None
decoder = None

latent_dim = 50  # Dimensionality of the latent space

img_width = 16
img_height = 16
num_channels = 1
input_shape = (img_height, img_width, num_channels)

input_img = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(input_img)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)

conv_shape = tf.keras.backend.int_shape(x)  # Shape of conv to be provided to decoder
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)

z_mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)  # Mean values of encoded input
z_sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

class CustomLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded) * img_width * img_height

        # KL divergence loss
        kl_loss = 1 + z_sigma - K.square(z_mu) - K.exp(z_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -.5

        return K.mean(recon_loss + (.75 * kl_loss))

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

n = 32  # Total number of rooms to generate and display
ROOM_SIZE = 16  # Assuming a room size of 16x16 tiles
tiles_per_row = 8  # Number of room tiles per row for display
num_categories = 9  # Since you have tiles 0 through 8

#Correction Algo:

bg_index = 0
wall_index = 1
bench_index = 2
table_index = 3
torch_index = 6
door_index = 7
crate_index = 8

def correct_edges(room):
    tile_types_to_convert_to_doors = {2,3,4, 5, 6}  # Tile types that can be converted to doors

    # Top and bottom edges
    for col in range(ROOM_SIZE - 1):
        # Ensure elements are treated as integers
        current_tile = int(room[0, col])
        next_tile = int(room[0, col + 1])
        if current_tile in tile_types_to_convert_to_doors and next_tile in tile_types_to_convert_to_doors:
            room[0, col] = door_index
            room[0, col + 1] = door_index
        current_tile = int(room[ROOM_SIZE - 1, col])
        next_tile = int(room[ROOM_SIZE - 1, col + 1])
        if current_tile in tile_types_to_convert_to_doors and next_tile in tile_types_to_convert_to_doors:
            room[ROOM_SIZE - 1, col] = door_index
            room[ROOM_SIZE - 1, col + 1] = door_index

    # Left and right edges
    for row in range(ROOM_SIZE - 1):
        current_tile = int(room[row, 0])
        next_tile = int(room[row + 1, 0])
        if current_tile in tile_types_to_convert_to_doors and next_tile in tile_types_to_convert_to_doors:
            room[row, 0] = door_index
            room[row + 1, 0] = door_index
        current_tile = int(room[row, ROOM_SIZE - 1])
        next_tile = int(room[row + 1, ROOM_SIZE - 1])
        if current_tile in tile_types_to_convert_to_doors and next_tile in tile_types_to_convert_to_doors:
            room[row, ROOM_SIZE - 1] = door_index
            room[row + 1, ROOM_SIZE - 1] = door_index

    # Set remaining non-door edge tiles to walls
    # Top and bottom edges
    for col in range(ROOM_SIZE):
        if col < ROOM_SIZE - 1 and room[0, col] != door_index:
            room[0, col] = wall_index
        if col < ROOM_SIZE - 1 and room[ROOM_SIZE - 1, col] != door_index:
            room[ROOM_SIZE - 1, col] = wall_index

    # Left and right edges
    for row in range(ROOM_SIZE):
        if row < ROOM_SIZE - 1 and room[row, 0] != door_index:
            room[row, 0] = wall_index
        if row < ROOM_SIZE - 1 and room[row, ROOM_SIZE - 1] != door_index:
            room[row, ROOM_SIZE - 1] = wall_index

    return room

def correct_crate_group(room):
    crate_index = 8

    # This function checks if the edges of a 4x4 box are "almost" all crates
    def is_almost_all_crates_4x4_center(row, col):
        edge_values = [
            room[row-1, col-1], room[row-1, col+2],  # Top corners
            room[row+2, col-1], room[row+2, col+2],  # Bottom corners
        ]
        edge_values.extend(room[row-1, col:col+2])  # Top edge
        edge_values.extend(room[row+2, col:col+2])  # Bottom edge
        edge_values.extend(room[row:row+2, col-1])  # Left edge
        edge_values.extend(room[row:row+2, col+2])  # Right edge
        crates_count = edge_values.count(crate_index)
        return crates_count >= 8  # All of the edge tiles should be crates

    # Check for any 4x4 area in the room that is not adjacent to the edge
    for row in range(1, ROOM_SIZE - 3):
        for col in range(1, ROOM_SIZE - 3):
            # If it's a 4x4 center and all edges are crates, update the center and edges
            if is_almost_all_crates_4x4_center(row, col):
                # Set center four tiles to either 0 or 4
                for i in range(row, row + 2):
                    for j in range(col, col + 2):
                        room[i, j] = random.choice([0, 4])
                # Set all the edges of the 4x4 box to 8
                room[row-1, col-1:col+3] = crate_index
                room[row+2, col-1:col+3] = crate_index
                room[row:row+2, col-1] = crate_index
                room[row:row+2, col+2] = crate_index
                print(f"4x4 box of crates detected and modified at center ({row+1}, {col+1})")

    return room

def correct_furniture_group(room):
    bench_index = 2
    table_index = 3
    visited = np.zeros_like(room, dtype=bool)  # To track visited cells for specific areas

    # Check bounds to ensure we don't go out of the array's limits
    def in_bounds(i, j):
        return 0 <= i < ROOM_SIZE and 0 <= j < ROOM_SIZE

    # Function to clear extra 2s and 3s around the box
    def clear_surrounding_area(row, col):
        for i in range(max(0, row-1), min(ROOM_SIZE, row+4)):
            for j in range(max(0, col-1), min(ROOM_SIZE, col+5)):
                if (i < row or i >= row+3) or (j < col or j >= col+4):
                    if room[i, j] == bench_index or room[i, j] == table_index:
                        room[i, j] = 0

    # checks if a 3x4 area is "almost" a table-bench combo
    def is_almost_table_bench_area(row, col):
        area_indices = [(row+i, col+j) for i in range(3) for j in range(4)]
        table_count = sum(room[r, c] == table_index for r, c in area_indices)
        bench_count = sum(room[r, c] == bench_index for r, c in area_indices)
        return (table_count + bench_count) >= 8

    # Function to convert an area to a table-bench combo and mark it visited
    def convert_to_table_bench(row, col):
        for i in range(3):
            for j in range(4):
                visited[row+i, col+j] = True
        room[row:row+3, col] = bench_index
        room[row:row+3, col+1:col+3] = table_index
        room[row:row+3, col+3] = bench_index

    # Iterate over the room to find and convert potential table-bench areas
    for row in range(1, ROOM_SIZE - 2):
        for col in range(1, ROOM_SIZE - 3):
            if is_almost_table_bench_area(row, col) and not visited[row, col]:
                clear_surrounding_area(row, col)
                convert_to_table_bench(row, col)

    # Clear all unvisited 2s and 3s
    for i in range(ROOM_SIZE):
        for j in range(ROOM_SIZE):
            if not visited[i, j] and (room[i, j] == bench_index or room[i, j] == table_index):
                room[i, j] = 0

    return room

def correct_furniture_group_old(room):
    bench_index = 2
    table_index = 3

    # check bounds to ensure we don't go out of the array's limits
    def in_bounds(i, j):
        return 0 <= i < ROOM_SIZE and 0 <= j < ROOM_SIZE

    # Function to clear extra 2s and 3s around the box
    def clear_surrounding_area(row, col):
        for i in range(max(0, row-1), min(ROOM_SIZE, row+4)):
            for j in range(max(0, col-1), min(ROOM_SIZE, col+5)):
                if (i < row or i >= row+3) or (j < col or j >= col+4):
                    if (room[i][j] != 1 and room[i][j] != 8 and room[i][j] != 7):
                        room[i][j] = 0

    # This function checks if a 3x4 area is "almost" a table-bench combo
    def is_almost_table_bench_area(row, col):
        # Define the 3x4 aRea indices
        area_indices = [(row+i, col+j) for i in range(3) for j in range(4)]

        # Count how many 2s and 3s are already present
        table_count = sum(room[r, c] == table_index for r, c in area_indices)
        bench_count = sum(room[r, c] == bench_index for r, c in area_indices)

        # more than 8 of 12
        return (table_count + bench_count) >= 8

    # Function to convert an area to a table-bench combo
    def convert_to_table_bench(row, col):
        # Set the first column to benches
        room[row:row+3, col] = bench_index

        # Set the second and third columns to tables
        room[row:row+3, col+1:col+3] = table_index

        # Set the fourth column to benches
        room[row:row+3, col+3] = bench_index
        #print(f"3x4 area converted to table-bench combo at starting cell ({row}, {col})")

    # Iterate over the room to find potential table-bench areas
    col = 1
    while col < ROOM_SIZE - 3 - 1:
        row = 1
        while row < ROOM_SIZE - 2 - 1:
            if is_almost_table_bench_area(row, col):
                clear_surrounding_area(row, col)
                convert_to_table_bench(row, col)
                row += 3  # Skip the next two rows to avoid rechecking the same area
            else:
                row += 1
        col += 1  # Move to the next column after checking all rows

    return room

def create_and_clear_boxes(room):
    target_indices = [5, 4, 6]

    # Helper function to clear adjacent tiles of a 1x4 box
    def clear_adjacent_box(start_row, start_col, vertical=False):
        if not vertical:
            if start_col - 1 >= 0 and room[start_row, start_col - 1] in target_indices:
                room[start_row, start_col - 1] = 0
            if start_col + 5 < ROOM_SIZE and room[start_row, start_col + 5] in target_indices:
                room[start_row, start_col + 5] = 0
        else:
            if start_row - 1 >= 0 and room[start_row - 1, start_col] in target_indices:
                room[start_row - 1, start_col] = 0
            if start_row + 5 < ROOM_SIZE and room[start_row + 5, start_col] in target_indices:
                room[start_row + 5, start_col] = 0

    # Check each row for a valid section of tiles
    for row in range(ROOM_SIZE):
        col = 0
        while col <= ROOM_SIZE - 4:
            section = room[row, col:col + 5]
            count_fives = np.count_nonzero(section == 5)
            if count_fives >= 2 and np.all(np.isin(section, target_indices)):
                room[row, col:col + 4] = 5  # Create a 1x4 box of '5's
                clear_adjacent_box(row, col)  # Clear adjacent 5, 4, or 6
                col += 3  # Skip past the new box
            col += 1

    # Check each column for a valid section of tiles
    for col in range(ROOM_SIZE):
        row = 0
        while row <= ROOM_SIZE - 4:
            section = room[row:row + 5, col]
            count_fives = np.count_nonzero(section == 5)
            if count_fives >= 2 and np.all(np.isin(section, target_indices)):
                room[row:row + 4, col] = 5
                clear_adjacent_box(row, col, vertical=True)
                row += 3
            row += 1

    return room

def correct_furniture_group_new(room):
    bench_index = 2
    table_index = 3

    # Check bounds to ensure we don't go out of the array's limits
    def in_bounds(i, j):
        return 0 <= i < ROOM_SIZE and 0 <= j < ROOM_SIZE

    # Function to clear extra 2s and 3s around the box
    def clear_surrounding_area(row, col):
        for i in range(max(0, row-1), min(ROOM_SIZE, row+4)):
            for j in range(max(0, col-1), min(ROOM_SIZE, col+5)):
                if (i < row or i >= row+3) or (j < col or j >= col+4):
                    if (room[i][j] != 1 and room[i][j] != 8 and room[i][j] != 7):
                        room[i][j] = 0

    # This function checks if a 3x4 area is "almost" a table-bench combo
    def is_almost_table_bench_area(row, col, isAll = False):
        # Define the 3x4 area indices
        area_indices = [(row+i, col+j) for i in range(3) for j in range(4)]


        if 0 <= (row+2) < ROOM_SIZE and 0 <= (col+3) < ROOM_SIZE:
            # Count how many 2s and 3s are already present
            table_count = sum(room[r, c] == table_index for r, c in area_indices)
            bench_count = sum(room[r, c] == bench_index for r, c in area_indices)

            # Define "almost" as at least 5 tables and 2 benches in the correct rows/columns
            if(isAll):
                return (table_count + bench_count) == 12
            else:
                return (table_count + bench_count) >= 8 #and wall_count <= 0
        else:
            return False

    # Function to convert an area to a table-bench combo
    def convert_to_table_bench(row, col):
        # Set the first column to benches
        room[row:row+3, col] = bench_index

        # Set the second and third columns to tables
        room[row:row+3, col+1:col+3] = table_index

        # Set the fourth column to benches
        room[row:row+3, col+3] = bench_index

    # Iterate over the room to find potential table-bench areas
    col = 1
    while col < ROOM_SIZE - 3 - 1:
        row = 1
        while row < ROOM_SIZE - 2 - 1:
            if is_almost_table_bench_area(row, col):
                clear_surrounding_area(row, col)
                convert_to_table_bench(row, col)
                row += 3  # Skip the next two rows to avoid rechecking the same area
            else:
                row += 1
        col += 1  # Move to the next column after checking all rows

    # Iterate again to cleanup
    col = 1
    while col < ROOM_SIZE - 1:
        row = 1
        while row < ROOM_SIZE - 1:
            if not is_almost_table_bench_area(row, col, True) and (room[row, col] == bench_index or room[row, col] == table_index):
                room[row, col] = bg_index
            row += 1
        col += 1

    return room

# Function to correct interior wall tiles to a different type
def correct_interior_walls(room):
    new_tile_index = 2  # The new tile type to replace interior walls

    # Iterate over the interior of the room
    for row in range(1, ROOM_SIZE - 1):
        for col in range(1, ROOM_SIZE - 1):
            if room[row, col] == wall_index:
                room[row, col] = new_tile_index

    return room

def delete_alone_tiles(room):
    updated_room = np.copy(room)

    # Check if a tile is surrounded only by walls and doors
    def is_alone(row, col):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROOM_SIZE and 0 <= nc < ROOM_SIZE:
                    if updated_room[nr, nc] == updated_room[row, col]:
                        return False
        return True

    # Iterate over the array to find "alone" tiles of 8s and 5s
    for row in range(ROOM_SIZE):
        for col in range(ROOM_SIZE):
            if room[row, col] == 5 or room[row, col] == 8:
                if is_alone(row, col):
                    room[row, col] = 0

    return room

def correct_doors(room):
    # Function to check if adjacent tiles include a wall or an edge
    def is_adjacent_to_wall(row, col):
        adjacent_positions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        return any((0 <= r < ROOM_SIZE and 0 <= c < ROOM_SIZE and room[r, c] == wall_index) or
                   r == 0 or r == ROOM_SIZE - 1 or c == 0 or c == ROOM_SIZE - 1
                   for r, c in adjacent_positions)

    # Iterate over the room, but not the edges
    for row in range(1, ROOM_SIZE - 1):
        for col in range(1, ROOM_SIZE - 1):
            if room[row, col] == door_index:
                # If the door is next to a wall, convert it to a torch; otherwise, a crate
                room[row, col] = torch_index if is_adjacent_to_wall(row, col) else crate_index

    return room

def load_model():
    global model
    global decoder
    model = tf.keras.models.load_model(model_path)
    decoder = keras.saving.load_model(model_path, custom_objects={'CustomLayer': CustomLayer()}, safe_mode=False, compile=True)

# Load the model when the application starts
load_model()

# Preprocess input vectors
preprocessed_vectors = np.random.normal(0, 1, size=(100, latent_dim))

# Precompute and cache model predictions for preprocessed vectors
cached_predictions = model.predict(preprocessed_vectors)
if cached_predictions.shape != (ROOM_SIZE, ROOM_SIZE):
    cached_predictions = cached_predictions.reshape((ROOM_SIZE, ROOM_SIZE))


def generate_image():

    # Apply correction algorithms
    corrected_image = correct_edges(cached_predictions)
    corrected_image = correct_interior_walls(corrected_image)
    corrected_image = correct_doors(corrected_image)
    corrected_image = correct_crate_group(corrected_image)
    corrected_image = correct_furniture_group(corrected_image)
    corrected_image = create_and_clear_boxes(corrected_image)
    corrected_image = delete_alone_tiles(corrected_image)
    # Additional corrections can be applied here if needed

    return corrected_image

@app.route("/predict2", methods=["POST"])
def predict():
     #data = {"success": False}

     if flask.request.method == "POST":
         image = generate_image()
         results = np.round(image[:,:,0], 3).tolist()

         return flask.jsonify(results)

     #return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='0.0.0.0')
