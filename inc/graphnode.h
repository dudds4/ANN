#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include <vector>

struct Layer;

struct GraphNode {

	bool activated = false;

	void setParent(Node* n);
	void setParents(const std::vector<Node*> &parentV);

private:
	Layer* layer;
	std::vector<GraphNode*> parents;
	std::vector<GraphNode*> children;

};

#endif