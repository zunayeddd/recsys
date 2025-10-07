// app.js
class SocialNetworkApp {
    constructor() {
        this.graphData = null;
        this.pageRankScores = null;
        this.selectedNode = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadDemoData());
        document.getElementById('computePR').addEventListener('click', () => this.computePageRank());
    }

    async loadDemoData() {
        try {
            const response = await fetch('data/karate.csv');
            const csvText = await response.text();
            this.graphData = this.parseCSV(csvText);
            
            document.getElementById('computePR').disabled = false;
            this.renderGraph();
            this.showNotification('Demo data loaded successfully!', 'success');
        } catch (error) {
            console.log(error)
            this.showNotification('Error loading demo data: ' + error.message, 'error');
        }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const edges = [];
        const nodes = new Set();

        // Skip header if exists, otherwise start from line 0
        const startLine = lines[0].includes('source') ? 1 : 0;
        
        for (let i = startLine; i < lines.length; i++) {
            const [source, target] = lines[i].split(',').map(val => val.trim());
            if (source && target) {
                edges.push({ source: parseInt(source), target: parseInt(target) });
                nodes.add(parseInt(source));
                nodes.add(parseInt(target));
            }
        }

        return {
            nodes: Array.from(nodes).map(id => ({ id })),
            edges: edges
        };
    }

    async computePageRank() {
        if (!this.graphData) {
            this.showNotification('Please load graph data first!', 'error');
            return;
        }

        try {
            this.showNotification('Computing PageRank...', 'info');
            
            // Convert graph to adjacency matrix format
            const { nodes, edges } = this.graphData;
            const nodeCount = nodes.length;
            const nodeIndexMap = new Map();
            nodes.forEach((node, index) => nodeIndexMap.set(node.id, index));

            // Create adjacency matrix
            const adjacencyMatrix = Array.from({ length: nodeCount }, () => 
                Array.from({ length: nodeCount }, () => 0)
            );

            edges.forEach(edge => {
                const sourceIdx = nodeIndexMap.get(edge.source);
                const targetIdx = nodeIndexMap.get(edge.target);
                adjacencyMatrix[sourceIdx][targetIdx] = 1;
                adjacencyMatrix[targetIdx][sourceIdx] = 1; // Undirected graph
            });

            // Compute PageRank
            this.pageRankScores = await computePageRankScores(adjacencyMatrix);
            
            // Update nodes with scores
            nodes.forEach((node, index) => {
                node.pagerank = this.pageRankScores[index];
            });

            this.updateTable();
            this.updateGraphVisualization();
            this.showNotification('PageRank computation completed!', 'success');
        } catch (error) {
            console.log(error)
            this.showNotification('Error computing PageRank: ' + error.message, 'error');
        }
    }

    updateTable() {
        const tableBody = document.getElementById('tableBody');
        tableBody.innerHTML = '';

        const { nodes, edges } = this.graphData;

        nodes.sort((a, b) => b.pagerank - a.pagerank);

        nodes.forEach(node => {
            const row = document.createElement('tr');
            row.dataset.nodeId = node.id;
            
            // Get friends for this node
            const friends = edges
                .filter(edge => edge.source === node.id || edge.target === node.id)
                .map(edge => edge.source === node.id ? edge.target : edge.source)
                .sort((a, b) => a - b);

            row.innerHTML = `
                <td>${node.id}</td>
                <td>${node.pagerank ? node.pagerank.toFixed(4) : 'N/A'}</td>
                <td>${friends.join(', ')}</td>
            `;

            row.addEventListener('click', () => this.selectNode(node.id));
            tableBody.appendChild(row);
        });
    }

    selectNode(nodeId) {
        this.selectedNode = nodeId;
        
        // Update table selection
        document.querySelectorAll('#tableBody tr').forEach(row => {
            row.classList.toggle('selected', parseInt(row.dataset.nodeId) === nodeId);
        });

        // Update graph selection
        if (window.graphVisualization) {
            window.graphVisualization.highlightNode(nodeId);
        }

        this.showNodeInfo(nodeId);
    }

    showNodeInfo(nodeId) {
        const nodeInfo = document.getElementById('nodeInfo');
        const currentFriends = document.getElementById('currentFriends');
        const recommendations = document.getElementById('recommendations');

        if (!this.graphData || !this.pageRankScores) {
            nodeInfo.style.display = 'none';
            return;
        }

        const { nodes, edges } = this.graphData;
        const node = nodes.find(n => n.id === nodeId);
        
        if (!node) return;

        // Get current friends
        const friends = edges
            .filter(edge => edge.source === nodeId || edge.target === nodeId)
            .map(edge => edge.source === nodeId ? edge.target : edge.source);

        currentFriends.innerHTML = `
            <strong>Current Friends (${friends.length}):</strong><br>
            ${friends.join(', ') || 'None'}
        `;

        // Get recommendations (top 3 non-friends by PageRank, excluding self)
        const nonFriends = nodes
            .filter(n => n.id !== nodeId && !friends.includes(n.id))
            .sort((a, b) => b.pagerank - a.pagerank)
            .slice(0, 3);

        if (nonFriends.length > 0) {
            recommendations.innerHTML = `
                <strong>Recommended Friends:</strong><br>
                ${nonFriends.map(n => 
                    `Node ${n.id} (Score: ${n.pagerank.toFixed(4)}) 
                    <button class="btn connect-btn" data-target="${n.id}">Connect</button>`
                ).join('<br>')}
            `;

            // Add event listeners to connect buttons
            document.querySelectorAll('.connect-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const targetNode = parseInt(btn.dataset.target);
                    this.addConnection(nodeId, targetNode);
                });
            });
        } else {
            recommendations.innerHTML = '<strong>No recommendations available</strong>';
        }

        nodeInfo.style.display = 'block';
    }

    addConnection(source, target) {
        if (!this.graphData) return;

        // Add new edge
        this.graphData.edges.push({ source, target });
        
        // Recompute PageRank
        this.computePageRank();
        this.showNotification(`Connected node ${source} to node ${target}`, 'success');
    }

    renderGraph() {
        if (!this.graphData) return;
        
        if (window.graphVisualization) {
            window.graphVisualization.destroy();
        }
        
        window.graphVisualization = new GraphVisualization(
            'graph',
            this.graphData,
            (nodeId) => this.selectNode(nodeId)
        );
    }

    updateGraphVisualization() {
        if (window.graphVisualization && this.graphData) {
            window.graphVisualization.updateData(this.graphData, this.pageRankScores);
        }
    }

    showNotification(message, type) {
        // Simple notification - in a real app you might want a more sophisticated system
        console.log(`${type.toUpperCase()}: ${message}`);
    }
}

//Initialize the application when the page loads
 document.addEventListener('DOMContentLoaded', () => {
     window.app = new SocialNetworkApp();
 });
