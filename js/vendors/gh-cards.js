let cards = document.getElementsByClassName('gh-card');

for(let card of cards) {
  let repo = card.getAttribute('data-repo');
  let url = 'https://api.github.com/repos/' + repo;
  
  fetch(url, {method: 'GET'}).then(resp => {
    return resp.json();
  }).then(json => {
    
    card.innerHTML = `
      <img class="gh" src="${card.getAttribute('data-image') || json.owner.avatar_url}">
      <div class="gh container">
        <h1 class="gh">
          <a class="gh" href="${json.html_url}">
            ${json.full_name}
          </a>
        </h1>
        <p class="gh">${json.description || ''}</p>
        <a class="gh" href="${json.html_url}/network">
          <i class="fa fa-fw fa-code-branch" aria-hidden="true"></i> ${json.forks_count}
        <a class="gh" href="${json.html_url}/stargazers">
          <i class="fa fa-fw fa-star" aria-hidden="true"></i> ${json.stargazers_count}
        <a class="gh" href="${json.html_url}/languages">
          <i class="fa fa-fw fa-terminal" aria-hidden="true"></i> ${json.language}
        </a>
      </div>
    `;
    
  }).catch(err => {
    console.log(err);
  });
}
