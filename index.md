---
title: Home
sections:
  - section_id: hero
    component: "hero_block.html"
    type: heroblock
    content: |-
      This is my portfolio
  - section_id: about
    component: "content_block.html"
    type: contentblock
    title: About me
    content: |-
      Ashli Update this section via the  index.md file in \content. This is the "about" excerpt. It can be used to provide a paragraph about yourself that people can read on the homepage to get a sense of who you are. There also exists a dedicated about page where you can write more about yourself for those who are interested.
    actions:
      - label: Contact
        url: "/contact"
  - section_id: recent-posts
    component: "posts_block.html"
    type: postsblock
    title: Featured Projects
    num_posts_displayed: 4
    actions:
      - label: View More Projects
        url: blog/index.html
menus:
  main:
    weight: 1
    title: Home
template: home
---
