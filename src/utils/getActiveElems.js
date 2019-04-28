import $ from 'jquery'

/**
 * 
 * @param {'Object'} ids An object with keys same as the keys in output varaible and properties id and thresh.
 */
const getActiveElems = (ids) => {
  const rootElem = $(':root')
  const scrollTop = rootElem.prop('scrollTop'),
    elems = {},
    active = {};

  for (const key in ids) {
    if (ids.hasOwnProperty(key)){
      elems[key] = $(`#${ids[key].id}`) || rootElem
    }
  }

  for (const key in elems) {
    if (elems.hasOwnProperty(key)) {
      active[key] = elems[key].offset().top - scrollTop < ids[key].thresh
    }
  }
  return active
}

export default getActiveElems