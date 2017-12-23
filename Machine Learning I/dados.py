import csv

def carregar_acessos():
	
	dados = []
	marcacoes = []

	arquivo = open ('buscas.csv', 'rb')
	leitor = csv.reader(arquivo)
	next(leitor)

	for home, como_funciona, contato, comprou in leitor:

		dados.append([int(home), int(como_funciona), int(contato)])
		marcacoes.append(int(comprou))

	return dados, marcacoes