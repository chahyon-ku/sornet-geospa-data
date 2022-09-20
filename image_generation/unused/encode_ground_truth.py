import numpy as np
from render_images import CONTAINER, SUPPORTER


DIRECTIONAL_RELATIONS = ['front', 'behind', 'right', 'left']
SPACIAL_RELATIONS = ['contain', 'support', 'can_contain', 'can_support']

RELATIONS = DIRECTIONAL_RELATIONS + SPACIAL_RELATIONS

RELATION_PHRASES = {
    'front': 'in front of',
    'behind': 'behind of',
    'right': 'right of',
    'left': 'left of',
    'contain': 'contains',
    'support': 'supports',
    'can_contain': 'can contain',
    'can_support': 'can support'
}


def create_relation_matrix(json_data) -> np.ndarray:
    ''' Takes the JSON data for a given scene, and returns the relation matrix just
    for that scene
    - json_data: the json data object returned by json.load()
    - returns: the (number_relations x number_objects x number_objects) matrix
               of relationships
    '''
    num_objs = len(json_data['objects'])
    rel_arr_shape = (len(RELATIONS), num_objs, num_objs)
    rel_arr = np.zeros(rel_arr_shape, dtype=float)

    # First front, back, right, left relations
    for rel_idx, (rel, rel_list) in enumerate(json_data['relationships'].items()):
        for first_obj_idx, other_idxs in enumerate(rel_list):
            rel_arr[rel_idx, first_obj_idx, other_idxs] = 1.0

    # Then contain and support relations
    # for obj_idx, obj_dict in enumerate(json_data['objects']):
    #     child_idx = obj_dict['child']
    #     if obj_dict['used'] == SUPPORTER:
    #         rel_arr[RELATIONS.index('support'), obj_idx, child_idx] = True
    #     elif obj_dict['used'] == CONTAINER:
    #         rel_arr[RELATIONS.index('contain'), obj_idx, child_idx] = True
    #
    # # Then can contain and can support
    # for rel in ['can_contain', 'can_support']:
    #     rel_list = json_data['relationships'][rel]
    #     for first_obj_idx, other_list in enumerate(rel_list):
    #         for other in other_list:
    #             idx = other['obj']
    #             val = other['val']
    #             rel_arr[RELATIONS.index(rel), first_obj_idx, idx] = val

    return rel_arr
