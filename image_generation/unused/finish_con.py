import numpy as np
import json
from render_images import SHAPE_TO_NUM
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pregenerate_predicates import TRUE, FALSE, UNCERTAIN

'''
Finish going through the uncertain '-1' values in the can contain matrix and 
manually fill in values
'''

CAN_CONTAIN_PATH = 'data/rels_can_contain.npy'


def main():
    print_instructions()
    can_contain_mat = finish_con()
    save_file(can_contain_mat, CAN_CONTAIN_PATH)


def print_instructions():
    # TODO: add a print statement with directions
    print('Options when prompted:')
    print('\ty - the objects can both fit in the container, store a 1 in the matrix')
    print('\tx - the objects can NOT both fit in the container, store a 0 in the matrix')
    print('\tnone - one of the current pairings (parent with child or other) cannot fit any more objects')


def finish_con() -> np.ndarray:
    ''' TODO: document this function
    - returns: updated numpy array representing the 3d containment relationships
    '''
    # Create figure to show images
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 15)
    ax[0].axis('off')
    ax[1].axis('off')
    fig.show()

    can_contain_mat = np.load(CAN_CONTAIN_PATH)
    if can_contain_mat.min() == UNCERTAIN: 
        print('can contain matrix has uncertain values')
    updates = 0
    save_frequency = 20
    shapes = get_shapes()
    for parent_shape, parent_shapeval in shapes:
        print(parent_shape)
        parent_num = SHAPE_TO_NUM[parent_shapeval]
        if can_contain_mat[parent_num].max() not in [TRUE, FALSE, UNCERTAIN]: 
            print('\tcan contain has unset values for parent')
        if can_contain_mat[parent_num].min() == UNCERTAIN: 
            print('\tcan contain has uncertain values for parent')
        for child_shape, child_shapeval in shapes:
            child_num = SHAPE_TO_NUM[child_shapeval]
            for other_shape, other_shapeval in shapes:
                other_num = SHAPE_TO_NUM[other_shapeval]
                # Sanity check can contain matrix values
                value: int = can_contain_mat[parent_num, child_num, other_num]
                assert(value in [TRUE, FALSE, UNCERTAIN]), f'unknown value for {parent_shape, child_shape, other_shape}: {value}'
                if value == UNCERTAIN:
                    # Relationship should be reflexive
                    if can_contain_mat[parent_num, other_num, child_num] != UNCERTAIN:
                        print(
                            f'reflexive already defined: {parent_shape} <- {child_shape}, {other_shape} : {can_contain_mat[parent_num][other_num][child_num]}')
                        can_contain_mat[parent_num, child_num,
                                        other_num] = can_contain_mat[parent_num][other_num][child_num]
                    else:
                        # Uncertain: display images to user and request input
                        one_pth = get_path(parent_shapeval, child_shapeval)
                        two_pth = get_path(parent_shapeval, other_shapeval)
                        one_img = mpimg.imread(one_pth)
                        two_img = mpimg.imread(two_pth)
                        # Show images
                        ax[0].imshow(one_img)
                        ax[0].set_title(f'child: {child_shapeval}')
                        ax[1].imshow(two_img)
                        ax[1].set_title(f'other: {other_shapeval}')
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        # Get user input
                        user = input(
                            'can contain both? (y, n)> ').strip().lower()
                        while user not in ['y', 'n', 'none']:
                            user = input(
                                'input invalid (y, n)> ').strip().lower()
                        if user == 'y':
                            can_contain_mat[parent_num,
                                            child_num, other_num] = 1
                            can_contain_mat[parent_num,
                                            other_num, child_num] = 1
                        elif user == 'none':
                            # Indicate that no other objects could fit with one of
                            # the current parent child parings, set entire
                            # vector for pair to zero
                            if child_shape == other_shape:
                                can_contain_mat[parent_num][child_num][:] = 0
                                can_contain_mat[parent_num][:][child_num] = 0
                                break
                            # Ask which shape
                            coro = input('child or other? > ').strip().lower()
                            while coro not in ['child', 'other']:
                                coro = input('invalid input. (child, other)> ')
                            if coro == 'child':
                                print(
                                    f'cannot contain any other objects: {parent_shape} <- {child_shape}')
                                can_contain_mat[parent_num, child_num, :] = 0
                                can_contain_mat[parent_num, :, child_num] = 0
                                break
                            else:
                                print(
                                    f'cannot contain any other objects: {parent_shape} <- {other_shape}')
                                can_contain_mat[parent_num, other_num, :] = 0
                                can_contain_mat[parent_num, :, other_num] = 0
                                break
                        else:
                            can_contain_mat[parent_num,
                                            child_num, other_num] = 0
                            can_contain_mat[parent_num,
                                            other_num, child_num] = 0
                    updates += 1
                    if updates > save_frequency:
                        # Get user input
                        uin = None
                        print(f'save updates to file? {CAN_CONTAIN_PATH}')
                        while uin not in ['y', 'n']:
                            uin = input('(y, n)>').strip().lower()
                        if uin == 'n':
                            print('ending process..')
                            exit(0)
                        save_file(can_contain_mat, CAN_CONTAIN_PATH)
                        updates = 0
    plt.close()
    return can_contain_mat


def get_shapes():
    ''' Loads the list of shapes from the json properties file and returns the 
    list of shapes from that file
    - returns: list of shapes
    '''
    with open('../data/properties/properties.json', 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
    return properties['shapes'].items()


def get_path(parent_shape: str, child_shape: str) -> str:
    ''' Takes the names of the parent and child shape, and returns the relative path
    to the top view of the containment relationship for those two objects
    - parent_shape: name of parent's shape (ex: cube_container_01)
    - child_shape: name of child's shape
    - returns: string path to the .png file
    '''
    return f'../output/relationships/{parent_shape}/contain/{child_shape}_top.png'


def save_file(matrix: np.ndarray, file_path: str):
    ''' Save the numpy array to a file 
    - matrix: the numpy array to save
    - file_path: the save location for the file
    '''
    print(f'saving {file_path}')
    np.save(file_path, matrix)


if __name__ == '__main__':
    main()
