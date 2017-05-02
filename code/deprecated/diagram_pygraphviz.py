import pygraphviz as p

#### IDEAS
# figure out how to lock down positions (left, input: right, output)
# change colors
# change cluster titles to stand out more
# change arrow labels to stand out less, also to match line clearly

##### global stuff
fontname = 'Helvetica'

incolor = 'lightslategrey'
prosp_color = 'darkorange'
spscolor = 'dodgerblue'
sampcolor = 'forestgreen'
outcolor = incolor

##### locations
inloc = [(10,6),(10,4)]
prosp_loc = [(7,5),(5,5)]

##### cluster attributes
attributes = {}
attributes['fontname'] = fontname
attributes['fontsize'] = 22
attributes['color'] = 'white'

###### edge attributes
edge_attr = {'dir':'forward'}#,'splines':'curved'}
edge_attr = {'dir':'forward','splines':'line'}


##### general graph properties
G = p.AGraph(directed=True,landscape='true',strict=False,ranksep='0.4')

#G.node_attr['color']='red'
#G.node_attr['style']='filled'
G.node_attr['shape']='box' # http://www.graphviz.org/doc/info/shapes.html
G.node_attr['fontname']=fontname


G.graph_attr['label']='Model Fitting Routine'

###### INPUTS
innodes = ['JMODEL','input photometry']
G.add_nodes_from(innodes,color=incolor)
attributes.update(fontcolor=incolor,
                  label='<<B>Inputs</B>>')

n=G.get_node(innodes[0])
#n.attr['pos']="%f,%f)"%inloc[0]
n=G.get_node(innodes[1])
#n.attr['pos']="%f,%f)"%inloc[1]

G1 = G.subgraph(nbunch=innodes,
                name="cluster1",**attributes)





###### Prospector
prosp_nodes = ['Prospector model','likelihood function']
G.add_nodes_from(prosp_nodes,color=prosp_color)
attributes.update(fontcolor=prosp_color,
                  label='<<B>Prospector</B>>')

n=G.get_node(prosp_nodes[0])
n.attr['pos']="%f,%f)"%prosp_loc[0]
n=G.get_node(prosp_nodes[1])
n.attr['pos']="%f,%f)"%prosp_loc[1]

G2 = G.subgraph(nbunch=prosp_nodes,
	            name="cluster2",**attributes)

## JMODEL fit + model parameters to Prospector model
G.add_edge(innodes[0],prosp_nodes[0],label='fit+model parameters',**edge_attr)

## JMODEL priors to likelihood
G.add_edge(innodes[0],prosp_nodes[1],label='priors',**edge_attr)

## Observed photometry to likelihood
G.add_edge(innodes[1],prosp_nodes[1],**edge_attr)

## between Prospectr model and likelihood
G.add_edge(prosp_nodes[0],prosp_nodes[1],label='photometry',**edge_attr)
G.add_edge(prosp_nodes[1],prosp_nodes[0],label='fit parameters',**edge_attr)




###### stellar pops code
spsnodes = ['python-fsps','FSPS']
G.add_nodes_from(spsnodes,color=spscolor)
attributes.update(fontcolor=spscolor,
                  label='<<B>SPS code</B>>')

G3 = G.subgraph(nbunch=spsnodes,
	            name="cluster3",**attributes)

# between python-fsps and FSPS
G.add_edge(spsnodes[0],spsnodes[1],label='model parameters',**edge_attr)
G.add_edge(spsnodes[1],spsnodes[0],label='model photometry',**edge_attr)

# between python-fsps and Prospectr model
G.add_edge(spsnodes[0],prosp_nodes[0],label='model photometry',**edge_attr)
G.add_edge(prosp_nodes[0],spsnodes[0],label='model parameters',**edge_attr)







###### sampler
sampnodes = ['emcee']
G.add_nodes_from(sampnodes,color=sampcolor)
attributes.update(fontcolor=sampcolor,
                  label='<<B>sampler</B>>')

G4 = G.subgraph(nbunch=sampnodes,
	            name="cluster4",**attributes)

# between sampler and likelihood
G.add_edge(prosp_nodes[1],sampnodes[0],label='likelihood',**edge_attr)
G.add_edge(sampnodes[0],prosp_nodes[1],label='fit parameters',**edge_attr)





###### Outputs
outnodes = ['fit parameter PDFs']
G.add_nodes_from(outnodes,color=outcolor)
attributes.update(fontcolor=outcolor,
                  label='<<B>outputs</B>>')

G5 = G.subgraph(nbunch=outnodes,
                name="cluster5",**attributes)

# from sampler to output
G.add_edge(sampnodes[0],outnodes[0],**edge_attr)

'''
G.draw('dot.png',prog="dot") # draw to png using circo
G.draw('neato.png',prog="neato") # draw to png using circo
G.draw('twopi.png',prog="twopi") # draw to png using circo
G.draw('circo.png',prog="circo") # draw to png using circo
'''
G.draw('fdp.png',prog="fdp",args="-s") # draw to png using circo
'''
G.draw('sfdp.png',prog="sfdp") # draw to png using circo
G.draw('tred.png',prog="tred") # draw to png using circo
G.draw('sccmap.png',prog="sccmap") # draw to png using circo
G.draw('ccomps.png',prog="ccomps") # draw to png using circo
G.draw('nop.png',prog="nop") # draw to png using circo
'''
