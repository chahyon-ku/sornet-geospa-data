import numpy

try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    print(e)


def get_vertex_color_layer(obj):
    vertex_color_layer = obj.data.vertex_colors.new()

    min_coord = numpy.zeros(3)
    max_coord = numpy.zeros(3)
    for loop_index, loop in enumerate(obj.data.loops):
        vertex_coord = obj.data.vertices[loop.vertex_index].co
        max_coord = numpy.maximum(max_coord, vertex_coord)
        min_coord = numpy.minimum(min_coord, vertex_coord)

    for loop_index, loop in enumerate(obj.data.loops):
        loop_vert_index = loop.vertex_index

        color = (numpy.array(obj.data.vertices[loop_vert_index].co) - min_coord) / (max_coord - min_coord)
        vertex_color_layer.data[loop_index].color[:3] = color
        vertex_color_layer.data[loop_index].color[3] = 1

    return vertex_color_layer


def get_nocs_material(vertex_color_layer_name):
    nocs_material = bpy.data.materials.new('nocs_material')
    nocs_material.use_nodes = True

    shader_node_vertex_color = nocs_material.node_tree.nodes.new('ShaderNodeVertexColor')
    shader_node_emission = nocs_material.node_tree.nodes.new('ShaderNodeEmission')
    material_output = nocs_material.node_tree.nodes.get('Material Output')

    shader_node_vertex_color.layer_name = vertex_color_layer_name

    nocs_material.node_tree.links.new(shader_node_vertex_color.outputs['Color'], shader_node_emission.inputs['Color'])
    nocs_material.node_tree.links.new(shader_node_emission.outputs['Emission'], material_output.inputs['Surface'])

    return nocs_material


def render_nocs():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if obj.name == 'Ground':
                emission_material = bpy.data.materials.new('emission_material')
                emission_material.use_nodes = True

                shader_node_emission = emission_material.node_tree.nodes.new('ShaderNodeEmission')
                material_output = emission_material.node_tree.nodes.get('Material Output')

                emission_material.node_tree.links.new(shader_node_emission.outputs['Emission'], material_output.inputs['Surface'])

                obj.data.materials.clear()
                obj.data.materials.append(emission_material)
                obj.active_material = emission_material
            else:
                vertex_color_layer = get_vertex_color_layer(obj)
                nocs_material = get_nocs_material(vertex_color_layer.name)

                obj.data.materials.clear()
                obj.data.materials.append(nocs_material)
                obj.active_material = nocs_material
            